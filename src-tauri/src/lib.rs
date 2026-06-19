use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, Read, Write},
    os::windows::process::CommandExt,
    path::Path,
    process::{Command, Stdio},
    sync::Mutex,
};
use tauri::{AppHandle, Emitter, Manager, State};

struct RunnerState {
    pid: Mutex<Option<u32>>,
    child: Mutex<Option<std::process::Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
    is_persistent: Mutex<bool>,
    active_api_stream: Mutex<Option<std::sync::mpsc::Sender<String>>>,
}

impl Default for RunnerState {
    fn default() -> Self {
        Self {
            pid: Mutex::new(None),
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            is_persistent: Mutex::new(false),
            active_api_stream: Mutex::new(None),
        }
    }
}

#[derive(Deserialize)]
struct GenerateRequest {
    runner_path: String,
    model_path: String,
    prompt: String,
    gpu_layers: u32,
    max_tokens: u32,
    max_steps: u32,
    #[serde(default)]
    is_diffusion: Option<bool>,
    #[serde(default)]
    cfg_scale: Option<f32>,
    #[serde(default)]
    t_min: Option<f32>,
    #[serde(default)]
    t_max: Option<f32>,
    #[serde(default)]
    entropy_bound: Option<f32>,
    #[serde(default)]
    stability: Option<u32>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Clone, Serialize)]
struct RunnerEvent {
    stream: String,
    line: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct DiffusionProgress {
    #[serde(default)]
    v: u32,
    step: u32,
    total_steps: u32,
    resolved: u32,
    total: u32,
    text: String,
    #[serde(default)]
    tokens: Option<Vec<String>>,
    #[serde(default)]
    token_types: Option<Vec<String>>,
    #[serde(default)]
    entropy: Option<Vec<Option<f32>>>,
    #[serde(default)]
    mean_entropy: Option<f32>,
    #[serde(default)]
    step_ms: Option<f32>,
}

#[derive(Serialize)]
struct SystemStatus {
    gpu_name: String,
    gpu_vendor: String,
    gpu_detected: bool,
    gpu_supports_metrics: bool,
    vram_used_mb: u32,
    vram_total_mb: u32,
    gpu_utilization: u32,
    temperature: u32,
    model_exists: bool,
    runner_exists: bool,
}

#[derive(Clone, Debug, Serialize)]
struct GpuDetection {
    name: String,
    vendor: String,
    detected: bool,
    supports_metrics: bool,
}

struct GpuCache {
    detection: Mutex<Option<GpuDetection>>,
}

impl Default for GpuCache {
    fn default() -> Self {
        Self {
            detection: Mutex::new(None),
        }
    }
}

#[derive(Serialize)]
struct DefaultPaths {
    runner_path: String,
    model_path: String,
    lora_dir: String,
}

#[derive(Deserialize)]
struct ImageRequest {
    runner_path: String,
    model_path: String,
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    #[serde(default)]
    guidance: f32,
    #[serde(default)]
    seed: u32,
    #[serde(default)]
    extra_args: String,
    #[serde(default)]
    lora_dir: String,
}

#[derive(Serialize)]
struct ImageResult {
    image_base64: String,
    output_path: String,
    duration_ms: f64,
}

#[derive(Clone, Serialize)]
struct ImageProgress {
    step: u32,
    total_steps: u32,
    message: String,
    preview_base64: Option<String>,
}

struct ImageRunnerState {
    pid: Mutex<Option<u32>>,
    running: Mutex<bool>,
}

impl Default for ImageRunnerState {
    fn default() -> Self {
        Self {
            pid: Mutex::new(None),
            running: Mutex::new(false),
        }
    }
}

#[derive(Debug, Serialize)]
struct GenerationResult {
    answer: String,
    thinking: Option<String>,
    duration_ms: Option<f64>,
}

fn emit_line(app: &AppHandle, stream: &str, line: String) {
    let _ = app.emit(
        "runner-output",
        RunnerEvent {
            stream: stream.to_string(),
            line,
        },
    );
}

fn strip_model_tokens(value: &str) -> String {
    value
        .replace("<|channel>thought", "")
        .replace("<|channel|>thought", "")
        .replace("<|channel>final", "")
        .replace("<|channel|>final", "")
        .replace("<channel|>", "")
        .replace("<|end|>", "")
        .replace("</s>", "")
        .trim()
        .to_string()
}

fn parse_duration_ms(raw: &str) -> Option<f64> {
    raw.lines().find_map(|line| {
        let timing = line.trim().strip_prefix("total time:")?.trim();
        timing
            .strip_suffix("ms")
            .unwrap_or(timing)
            .split(',')
            .next()?
            .trim()
            .parse()
            .ok()
    })
}

fn parse_model_response(raw: &str) -> GenerationResult {
    let without_timing = raw
        .lines()
        .take_while(|line| !line.trim_start().starts_with("total time:"))
        .collect::<Vec<_>>()
        .join("\n");

    let split = if let Some(index) = without_timing.rfind("<channel|>") {
        Some((index, "<channel|>".len()))
    } else if let Some(index) = without_timing.rfind("<|channel>final") {
        Some((index, "<|channel>final".len()))
    } else if let Some(index) = without_timing.rfind("<|channel|>final") {
        Some((index, "<|channel|>final".len()))
    } else {
        None
    };

    let (thinking, answer) = if let Some((index, marker_length)) = split {
        let thought = strip_model_tokens(&without_timing[..index]);
        let answer = strip_model_tokens(&without_timing[index + marker_length..]);
        ((!thought.is_empty()).then_some(thought), answer)
    } else {
        (None, strip_model_tokens(&without_timing))
    };

    GenerationResult {
        answer,
        thinking,
        duration_ms: parse_duration_ms(raw),
    }
}

#[tauri::command]
async fn generate(
    app: AppHandle,
    request: GenerateRequest,
) -> Result<GenerationResult, String> {
    if !Path::new(&request.runner_path).is_file() {
        return Err(format!("Runner no encontrado: {}", request.runner_path));
    }
    if !Path::new(&request.model_path).is_file() {
        return Err(format!("Modelo no encontrado: {}", request.model_path));
    }

    let app_handle = app.clone();
    {
        let runner_state = app.state::<RunnerState>();
        let pid_slot = runner_state.pid.lock().map_err(|error| error.to_string())?;
        if pid_slot.is_some() {
            return Err("Ya existe una generación activa".to_string());
        }
    }

    tauri::async_runtime::spawn_blocking(move || {
        let mut args = vec![
            "-m".to_string(),
            request.model_path.clone(),
            "-ngl".to_string(),
            request.gpu_layers.to_string(),
            "-n".to_string(),
            request.max_tokens.to_string(),
            "-p".to_string(),
            request.prompt.clone(),
        ];

        if request.is_diffusion.unwrap_or(true) {
            args.push("--diffusion-eb-max-steps".to_string());
            args.push(request.max_steps.to_string());

            if let Some(cfg) = request.cfg_scale {
                args.push("--diffusion-cfg-scale".to_string());
                args.push(cfg.to_string());
            }
            if let Some(t_min) = request.t_min {
                args.push("--diffusion-eb-t-min".to_string());
                args.push(t_min.to_string());
            }
            if let Some(t_max) = request.t_max {
                args.push("--diffusion-eb-t-max".to_string());
                args.push(t_max.to_string());
            }
            if let Some(eb_bound) = request.entropy_bound {
                args.push("--diffusion-eb-entropy-bound".to_string());
                args.push(eb_bound.to_string());
            }
            if let Some(stab) = request.stability {
                args.push("--diffusion-eb-stability".to_string());
                args.push(stab.to_string());
            }
            if let Some(conf) = request.confidence {
                args.push("--diffusion-eb-confidence".to_string());
                args.push(conf.to_string());
            }
        }

        let mut child = Command::new(&request.runner_path)
            .args(&args)
            .env("DIFFUSION_APP_PROGRESS", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .creation_flags(0x08000000)
            .spawn()
            .map_err(|error| format!("No se pudo iniciar el runner: {error}"))?;

        let pid = child.id();
        if let Ok(mut active_pid) = app_handle.state::<RunnerState>().pid.lock() {
            *active_pid = Some(pid);
        }
        emit_line(&app_handle, "system", format!("Proceso iniciado (PID {pid})"));

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let output = std::sync::Arc::new(Mutex::new(String::new()));

        let stdout_thread = stdout.map(|stream| {
            let app = app_handle.clone();
            let captured = output.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if let Ok(mut text) = captured.lock() {
                        text.push_str(&line);
                        text.push('\n');
                    }
                    emit_line(&app, "stdout", line);
                }
            })
        });

        let stderr_thread = stderr.map(|stream| {
            let app = app_handle.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if let Some(payload) = line.strip_prefix("DG_PROGRESS\t") {
                        if let Ok(progress) = serde_json::from_str::<DiffusionProgress>(payload) {
                            let _ = app.emit("diffusion-progress", progress);
                            continue;
                        }
                    }

                    emit_line(&app, "stderr", line);
                }
            })
        });

        let status = child.wait().map_err(|error| error.to_string())?;
        if let Some(thread) = stdout_thread {
            let _ = thread.join();
        }
        if let Some(thread) = stderr_thread {
            let _ = thread.join();
        }
        if let Ok(mut active_pid) = app_handle.state::<RunnerState>().pid.lock() {
            *active_pid = None;
        }
        emit_line(
            &app_handle,
            "system",
            format!("Proceso finalizado con código {}", status.code().unwrap_or(-1)),
        );

        let text = output.lock().map_err(|error| error.to_string())?.clone();
        if status.success() {
            let result = parse_model_response(&text);
            if result.answer.is_empty() {
                Err("El modelo terminó sin una respuesta final legible.".to_string())
            } else {
                Ok(result)
            }
        } else {
            Err(format!(
                "El runner terminó con código {}. Revisa el registro.",
                status.code().unwrap_or(-1)
            ))
        }
    })
    .await
    .map_err(|error| error.to_string())?
}

#[cfg(test)]
mod tests {
    use super::parse_model_response;

    #[test]
    fn separates_reasoning_answer_and_timing() {
        let raw = "<|channel>thought\nanálisis privado\n<channel|>Hola, ¿en qué puedo ayudarte?\ntotal time: 16574.02ms";
        let result = parse_model_response(raw);
        assert_eq!(result.thinking.as_deref(), Some("análisis privado"));
        assert_eq!(result.answer, "Hola, ¿en qué puedo ayudarte?");
        assert_eq!(result.duration_ms, Some(16574.02));
    }

    #[test]
    fn preserves_plain_responses() {
        let result = parse_model_response("Respuesta normal.\n");
        assert_eq!(result.answer, "Respuesta normal.");
        assert_eq!(result.thinking, None);
        assert_eq!(result.duration_ms, None);
    }
}

#[tauri::command]
fn get_default_paths(app: AppHandle) -> DefaultPaths {
    let bundled_runner = app
        .path()
        .resource_dir()
        .ok()
        .map(|path| path.join("runtime").join("llama-diffusion-cli.exe"))
        .filter(|path| path.is_file());
    let development_runner = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("runtime")
        .join("llama-diffusion-cli.exe");

    let home_dir = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok();
    let default_model_path = home_dir.as_ref().map(|home| {
        Path::new(home)
            .join(".lmstudio")
            .join("models")
            .join("google")
            .join("diffusiongemma-26b-a4b-it-GGUF")
            .join("diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
    });

    let model_path = if let Some(ref path) = default_model_path {
        if path.is_file() {
            path.to_string_lossy().into_owned()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    DefaultPaths {
        runner_path: bundled_runner
            .unwrap_or(development_runner)
            .to_string_lossy()
            .into_owned(),
        model_path,
        lora_dir: String::new(),
    }
}

#[tauri::command]
fn stop_generation(state: State<'_, RunnerState>) -> Result<(), String> {
    let pid = *state.pid.lock().map_err(|error| error.to_string())?;
    if let Some(pid) = pid {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(0x08000000)
            .status();
    }

    {
        if let Ok(mut pid_guard) = state.pid.lock() {
            *pid_guard = None;
        }
        if let Ok(mut child_guard) = state.child.lock() {
            *child_guard = None;
        }
        if let Ok(mut stdin_guard) = state.stdin.lock() {
            *stdin_guard = None;
        }
        if let Ok(mut is_p) = state.is_persistent.lock() {
            *is_p = false;
        }
    }
    Ok(())
}

#[tauri::command]
async fn start_model(
    app: AppHandle,
    state: State<'_, RunnerState>,
    runner_path: String,
    model_path: String,
    gpu_layers: u32,
    max_tokens: u32,
    cfg_scale: Option<f32>,
    t_min: Option<f32>,
    t_max: Option<f32>,
    entropy_bound: Option<f32>,
    stability: Option<u32>,
    confidence: Option<f32>,
) -> Result<(), String> {
    if !Path::new(&runner_path).is_file() {
        return Err(format!("Runner no encontrado: {}", runner_path));
    }
    if !Path::new(&model_path).is_file() {
        return Err(format!("Modelo no encontrado: {}", model_path));
    }

    // Verificar si ya está iniciado
    {
        let child_guard = state.child.lock().map_err(|e| e.to_string())?;
        if child_guard.is_some() {
            return Err("El modelo ya está iniciado".to_string());
        }
    }

    let app_handle = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let state_handle = app_handle.state::<RunnerState>();
        let mut args = vec![
            "-m".to_string(),
            model_path.clone(),
            "-ngl".to_string(),
            gpu_layers.to_string(),
            "-n".to_string(),
            max_tokens.to_string(),
            "-cnv".to_string(),
        ];

        if let Some(cfg) = cfg_scale {
            args.push("--diffusion-cfg-scale".to_string());
            args.push(cfg.to_string());
        }
        if let Some(t_min) = t_min {
            args.push("--diffusion-eb-t-min".to_string());
            args.push(t_min.to_string());
        }
        if let Some(t_max) = t_max {
            args.push("--diffusion-eb-t-max".to_string());
            args.push(t_max.to_string());
        }
        if let Some(eb_bound) = entropy_bound {
            args.push("--diffusion-eb-entropy-bound".to_string());
            args.push(eb_bound.to_string());
        }
        if let Some(stab) = stability {
            args.push("--diffusion-eb-stability".to_string());
            args.push(stab.to_string());
        }
        if let Some(conf) = confidence {
            args.push("--diffusion-eb-confidence".to_string());
            args.push(conf.to_string());
        }

        let mut child = Command::new(&runner_path)
            .args(&args)
            .env("DIFFUSION_APP_PROGRESS", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::piped())
            .creation_flags(0x08000000)
            .spawn()
            .map_err(|error| format!("No se pudo iniciar el runner: {error}"))?;

        let pid = child.id();
        let stdin = child.stdin.take().ok_or("No se pudo abrir stdin")?;
        let stdout = child.stdout.take().ok_or("No se pudo abrir stdout")?;
        let stderr = child.stderr.take().ok_or("No se pudo abrir stderr")?;

        {
            if let Ok(mut pid_guard) = state_handle.pid.lock() {
                *pid_guard = Some(pid);
            }
            if let Ok(mut child_guard) = state_handle.child.lock() {
                *child_guard = Some(child);
            }
            if let Ok(mut stdin_guard) = state_handle.stdin.lock() {
                *stdin_guard = Some(stdin);
            }
            if let Ok(mut is_p) = state_handle.is_persistent.lock() {
                *is_p = true;
            }
        }

        emit_line(&app_handle, "system", format!("Modelo cargado en memoria (PID {pid})"));
        let _ = app_handle.emit("model-status-changed", "ready");

        // Hilo de escucha stdout (No bloqueante para prompts interactivos)
        let app_stdout = app_handle.clone();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut buffer = Vec::new();
            let mut byte_buf = [0; 1];
            
            while let Ok(n) = reader.read(&mut byte_buf) {
                if n == 0 {
                    break; // EOF
                }
                let byte = byte_buf[0];
                buffer.push(byte);
                
                if byte == b'\n' {
                    if let Ok(line) = String::from_utf8(buffer.clone()) {
                        emit_line(&app_stdout, "stdout", line.clone());
                        if let Ok(guard) = app_stdout.state::<RunnerState>().active_api_stream.lock() {
                            if let Some(ref sender) = *guard {
                                let _ = sender.send(line);
                            }
                        }
                    }
                    buffer.clear();
                } else if buffer.ends_with(b"\n> ") || buffer == b"> " || buffer.ends_with(b"\r\n> ") {
                    let line_raw = String::from_utf8(buffer.clone()).unwrap_or_default();
                    let line = line_raw.trim_end_matches("> ").trim_end_matches("\n").trim_end_matches("\r").to_string();
                    emit_line(&app_stdout, "stdout", line_raw);

                    if let Ok(guard) = app_stdout.state::<RunnerState>().active_api_stream.lock() {
                        if let Some(ref sender) = *guard {
                            let _ = sender.send(line);
                        }
                    }
                    buffer.clear();
                    let _ = app_stdout.emit("generation-finished", ());

                    // Cerrar el canal
                    if let Ok(mut guard) = app_stdout.state::<RunnerState>().active_api_stream.lock() {
                        *guard = None;
                    }
                }
            }
        });

        // Hilo de escucha stderr
        let app_stderr = app_handle.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Some(payload) = line.strip_prefix("DG_PROGRESS\t") {
                    if let Ok(progress) = serde_json::from_str::<DiffusionProgress>(payload) {
                        let _ = app_stderr.emit("diffusion-progress", progress);
                        continue;
                    }
                }

                emit_line(&app_stderr, "stderr", line);
            }
        });

        Ok::<(), String>(())
    })
    .await
    .map_err(|error| error.to_string())??;

    Ok(())
}

#[tauri::command]
fn stop_model(app: AppHandle, state: State<'_, RunnerState>) -> Result<(), String> {
    let pid = *state.pid.lock().map_err(|error| error.to_string())?;
    if let Some(pid) = pid {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(0x08000000)
            .status();
    }

    {
        if let Ok(mut pid_guard) = state.pid.lock() {
            *pid_guard = None;
        }
        if let Ok(mut child_guard) = state.child.lock() {
            *child_guard = None;
        }
        if let Ok(mut stdin_guard) = state.stdin.lock() {
            *stdin_guard = None;
        }
        if let Ok(mut is_p) = state.is_persistent.lock() {
            *is_p = false;
        }
    }

    emit_line(&app, "system", "Modelo liberado de memoria".to_string());
    let _ = app.emit("model-status-changed", "stopped");

    Ok(())
}

#[tauri::command]
fn send_interactive_prompt(state: State<'_, RunnerState>, prompt: String) -> Result<(), String> {
    let mut stdin_guard = state.stdin.lock().map_err(|error| error.to_string())?;
    if let Some(ref mut stdin) = *stdin_guard {
        use std::io::Write;
        writeln!(stdin, "{}", prompt).map_err(|error| error.to_string())?;
        stdin.flush().map_err(|error| error.to_string())?;
        Ok(())
    } else {
        Err("El modelo no está cargado".to_string())
    }
}

#[tauri::command]
fn is_model_loaded(state: State<'_, RunnerState>) -> bool {
    state.child.lock().map(|child| child.is_some()).unwrap_or(false)
}

fn infer_vendor(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.contains("nvidia")
        || lower.contains("geforce")
        || lower.contains("rtx")
        || lower.contains("gtx")
        || lower.contains("quadro")
        || lower.contains("tesla")
    {
        "NVIDIA".to_string()
    } else if lower.contains("amd")
        || lower.contains("radeon")
        || lower.contains(" rx ")
        || lower.contains("r9")
        || lower.contains("r7")
    {
        "AMD".to_string()
    } else if lower.contains("intel")
        || lower.contains("arc ")
        || lower.contains("iris")
        || lower.contains("uhd")
        || lower.contains("hd graphics")
    {
        "Intel".to_string()
    } else {
        "Desconocido".to_string()
    }
}

fn detect_nvidia_name() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader,nounits"])
        .creation_flags(0x08000000)
        .output()
        .ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    let name = stdout.trim().to_string();
    if name.is_empty()
        || name.to_lowercase().contains("not found")
        || name.to_lowercase().contains("no devices")
        || name.to_lowercase().contains("command not found")
    {
        return None;
    }
    Some(name)
}

fn query_nvidia_metrics() -> (u32, u32, u32, u32) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .creation_flags(0x08000000)
        .output();
    let values = output
        .ok()
        .and_then(|result| String::from_utf8(result.stdout).ok())
        .unwrap_or_default();
    let parts: Vec<&str> = values.trim().split(',').map(str::trim).collect();
    (
        parts.first().and_then(|v| v.parse().ok()).unwrap_or(0),
        parts.get(1).and_then(|v| v.parse().ok()).unwrap_or(0),
        parts.get(2).and_then(|v| v.parse().ok()).unwrap_or(0),
        parts.get(3).and_then(|v| v.parse().ok()).unwrap_or(0),
    )
}

fn detect_gpu_wmi_windows() -> Option<String> {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Sort-Object AdapterRAM -Descending | Select-Object -ExpandProperty Name",
        ])
        .creation_flags(0x08000000)
        .output()
        .ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    stdout
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .filter(|l| {
            let lower = l.to_lowercase();
            !lower.contains("microsoft basic")
                && !lower.contains("remote desktop")
                && !lower.contains("vmware")
                && !lower.contains("virtualbox")
                && !lower.contains("hyper-v")
                && !lower.contains("display")
                && !lower.contains("mirror")
        })
        .next()
        .map(|s| s.to_string())
}

fn detect_gpu() -> GpuDetection {
    // 1. NVIDIA via nvidia-smi (full metrics available)
    if let Some(name) = detect_nvidia_name() {
        return GpuDetection {
            name,
            vendor: "NVIDIA".to_string(),
            detected: true,
            supports_metrics: true,
        };
    }

    // 2. Windows: WMI fallback for any vendor (name only, no live metrics)
    if let Some(name) = detect_gpu_wmi_windows() {
        let vendor = infer_vendor(&name);
        return GpuDetection {
            name,
            vendor,
            detected: true,
            supports_metrics: false,
        };
    }

    // 3. No GPU detected
    GpuDetection {
        name: "Sin GPU dedicada".to_string(),
        vendor: "—".to_string(),
        detected: false,
        supports_metrics: false,
    }
}

fn resolve_gpu_detection(gpu_cache: &GpuCache) -> GpuDetection {
    let mut cache = gpu_cache.detection.lock().unwrap();
    if let Some(ref detection) = *cache {
        return detection.clone();
    }
    let detection = detect_gpu();
    *cache = Some(detection.clone());
    detection
}

#[tauri::command]
fn refresh_gpu_detection(gpu_cache: State<'_, GpuCache>) -> Result<GpuDetection, String> {
    let detection = detect_gpu();
    if let Ok(mut cache) = gpu_cache.detection.lock() {
        *cache = Some(detection.clone());
    }
    Ok(detection)
}

#[tauri::command]
fn get_system_status(
    runner_path: String,
    model_path: String,
    gpu_cache: State<'_, GpuCache>,
) -> SystemStatus {
    let detection = resolve_gpu_detection(&gpu_cache);

    let (vram_used_mb, vram_total_mb, gpu_utilization, temperature) =
        if detection.supports_metrics {
            query_nvidia_metrics()
        } else {
            (0, 0, 0, 0)
        };

    SystemStatus {
        gpu_name: detection.name,
        gpu_vendor: detection.vendor,
        gpu_detected: detection.detected,
        gpu_supports_metrics: detection.supports_metrics,
        vram_used_mb,
        vram_total_mb,
        gpu_utilization,
        temperature,
        model_exists: Path::new(&model_path).is_file(),
        runner_exists: Path::new(&runner_path).is_file(),
    }
}

fn parse_image_progress(line: &str, preview_path: &str) -> Option<ImageProgress> {
    let trimmed = line.trim();
    // formatos comunes: "step 5/30", "5/30", " 50%|...| 15/30"
    if let Some(caps) = regex_simple_step(trimmed) {
        let preview_base64 = if std::path::Path::new(preview_path).is_file() {
            if let Ok(bytes) = std::fs::read(preview_path) {
                use base64::{engine::general_purpose::STANDARD, Engine as _};
                Some(STANDARD.encode(&bytes))
            } else {
                None
            }
        } else {
            None
        };

        return Some(ImageProgress {
            step: caps.0,
            total_steps: caps.1,
            message: trimmed.to_string(),
            preview_base64,
        });
    }
    None
}

fn regex_simple_step(s: &str) -> Option<(u32, u32)> {
    // busca "N/M" donde N y M son números
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            let n: u32 = s[start..i].parse().ok()?;
            if i < bytes.len() && bytes[i] == b'/' {
                i += 1;
                let m_start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
                if i > m_start {
                    let m: u32 = s[m_start..i].parse().ok()?;
                    if m > 0 && n <= m {
                        return Some((n, m));
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

#[tauri::command]
async fn generate_image(
    app: AppHandle,
    request: ImageRequest,
) -> Result<ImageResult, String> {
    if !Path::new(&request.runner_path).is_file() {
        return Err(format!("Runner de imagen no encontrado: {}", request.runner_path));
    }
    if !Path::new(&request.model_path).is_file() {
        return Err(format!("Modelo de imagen no encontrado: {}", request.model_path));
    }

    {
        let state = app.state::<ImageRunnerState>();
        let running = state.running.lock().map_err(|e| e.to_string())?;
        if *running {
            return Err("Ya hay una generación de imagen en curso".to_string());
        }
    }
    {
        let state = app.state::<ImageRunnerState>();
        let mut running = state.running.lock().map_err(|e| e.to_string())?;
        *running = true;
    }

    let app_handle = app.clone();
    let output_path = std::env::temp_dir()
        .join(format!("lms_image_{}.png", std::process::id()))
        .to_string_lossy()
        .to_string();

    let preview_path = std::env::temp_dir()
        .join(format!("lms_preview_{}.png", std::process::id()))
        .to_string_lossy()
        .to_string();

    let preview_path_for_args = preview_path.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = Command::new(&request.runner_path);
        cmd.args([
            "-m", &request.model_path,
            "-p", &request.prompt,
            "-W", &request.width.to_string(),
            "-H", &request.height.to_string(),
            "--steps", &request.steps.to_string(),
            "-o", &output_path,
            "--preview-path", &preview_path_for_args,
            "--preview", "vae",
        ]);

        if request.guidance > 0.0 {
            cmd.args(["--cfg-scale", &request.guidance.to_string()]);
        }
        if request.seed > 0 {
            cmd.args(["--seed", &request.seed.to_string()]);
        }
        if !request.negative_prompt.is_empty() {
            cmd.args(["-n", &request.negative_prompt]);
        }
        if !request.lora_dir.is_empty() {
            cmd.args(["--lora-model-dir", &request.lora_dir]);
        }
        if !request.extra_args.is_empty() {
            for arg in request.extra_args.split_whitespace() {
                cmd.arg(arg);
            }
        }

        cmd.env("DIFFUSION_APP_PROGRESS", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .creation_flags(0x08000000);

        let start = std::time::Instant::now();

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("No se pudo iniciar el runner de imagen: {e}"))?;

        let pid = child.id();
        {
            let state = app_handle.state::<ImageRunnerState>();
            let mut pid_lock = state.pid.lock().map_err(|e| e.to_string())?;
            *pid_lock = Some(pid);
        }
        emit_line(&app_handle, "system", format!("Generación de imagen iniciada (PID {pid})"));

        let stderr = child.stderr.take();
        if let Some(stderr) = stderr {
            let app_clone = app_handle.clone();
            let preview_path_clone = preview_path.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    let line = line.trim_end();
                    if !line.is_empty() {
                        emit_line(&app_clone, "stderr", line.to_string());
                        if let Some(prog) = parse_image_progress(line, &preview_path_clone) {
                            let _ = app_clone.emit("image-progress", prog);
                        }
                    }
                }
            });
        }

        let stdout = child.stdout.take();
        if let Some(stdout) = stdout {
            let app_clone = app_handle.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    emit_line(&app_clone, "stdout", line);
                }
            });
        }

        let status = child
            .wait()
            .map_err(|e| format!("Error esperando al runner: {e}"))?;

        {
            let state = app_handle.state::<ImageRunnerState>();
            let mut pid_lock = state.pid.lock().map_err(|e| e.to_string())?;
            *pid_lock = None;
        }

        if !status.success() {
            return Err(format!("El runner terminó con código {}", status.code().unwrap_or(-1)));
        }

        let image_bytes = std::fs::read(&output_path)
            .map_err(|e| format!("No se pudo leer la imagen de salida: {e}"))?;

        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let b64 = STANDARD.encode(&image_bytes);

        // Limpiar archivos temporales
        let _ = std::fs::remove_file(&output_path);
        let _ = std::fs::remove_file(&preview_path);

        Ok(ImageResult {
            image_base64: b64,
            output_path,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    })
    .await
    .map_err(|e| format!("Error en spawn_blocking: {e}"))?;

    {
        let state = app.state::<ImageRunnerState>();
        let mut running = state.running.lock().map_err(|e| e.to_string())?;
        *running = false;
    }

    let _ = app.emit("image-generation-finished", ());
    result
}

#[tauri::command]
fn stop_image_generation(state: State<'_, ImageRunnerState>) -> Result<(), String> {
    let pid_lock = state.pid.lock().map_err(|e| e.to_string())?;
    if let Some(pid) = *pid_lock {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(0x08000000)
            .output();
    }
    drop(pid_lock);
    let mut running = state.running.lock().map_err(|e| e.to_string())?;
    *running = false;
    Ok(())
}

#[tauri::command]
fn is_image_generating(state: State<'_, ImageRunnerState>) -> bool {
    state.running.lock().map(|r| *r).unwrap_or(false)
}

#[tauri::command]
fn get_default_image_paths(app: AppHandle) -> DefaultPaths {
    let bundled_runner = app
        .path()
        .resource_dir()
        .ok()
        .map(|path| path.join("runtime").join("sd-cli.exe"))
        .map(|p| p.to_string_lossy().to_string());

    let development_runner = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("runtime").join("sd-cli.exe"))
        .map(|p| p.to_string_lossy().to_string());

    let home_dir = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok();
    let default_model_path = home_dir.as_ref().map(|home| {
        Path::new(home)
            .join("Documents")
            .join("ComfyUI")
            .join("models")
            .join("checkpoints")
            .join("ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    });
    let default_lora_dir = home_dir.as_ref().map(|home| {
        Path::new(home)
            .join("Documents")
            .join("ComfyUI")
            .join("models")
            .join("loras")
    });

    let model_path = if let Some(ref path) = default_model_path {
        if path.is_file() {
            path.to_string_lossy().into_owned()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let lora_dir = if let Some(ref path) = default_lora_dir {
        if path.is_dir() {
            path.to_string_lossy().into_owned()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    DefaultPaths {
        runner_path: bundled_runner.unwrap_or(development_runner.unwrap_or_default()),
        model_path,
        lora_dir,
    }
}

#[derive(Serialize)]
struct ParsedMetadata {
    prompt: String,
    negative_prompt: String,
    steps: Option<u32>,
    guidance: Option<f32>,
    seed: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    model: Option<String>,
}

fn read_png_parameters(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 8 || &bytes[0..8] != [137, 80, 78, 71, 13, 10, 26, 10] {
        return None;
    }

    let mut pos = 8;
    while pos + 12 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[pos..pos+4].try_into().ok()?) as usize;
        let chunk_type = &bytes[pos+4..pos+8];
        
        if chunk_type == b"tEXt" {
            let chunk_data = &bytes[pos+8..pos+8+length];
            if let Some(null_pos) = chunk_data.iter().position(|&b| b == 0) {
                if let Ok(keyword) = std::str::from_utf8(&chunk_data[..null_pos]) {
                    if keyword == "parameters" {
                        if let Ok(value) = std::str::from_utf8(&chunk_data[null_pos+1..]) {
                            return Some(value.to_string());
                        }
                    }
                }
            }
        } else if chunk_type == b"iTXt" {
            let chunk_data = &bytes[pos+8..pos+8+length];
            if let Some(kw_null) = chunk_data.iter().position(|&b| b == 0) {
                if let Ok(keyword) = std::str::from_utf8(&chunk_data[..kw_null]) {
                    if keyword == "parameters" {
                        let comp_flag = chunk_data[kw_null + 1];
                        let start = kw_null + 3;
                        if start < chunk_data.len() {
                            let rem = &chunk_data[start..];
                            if let Some(lang_null) = rem.iter().position(|&b| b == 0) {
                                let rem2 = &rem[lang_null + 1..];
                                if let Some(trans_null) = rem2.iter().position(|&b| b == 0) {
                                    let text_bytes = &rem2[trans_null + 1..];
                                    if comp_flag == 0 {
                                        if let Ok(text) = std::str::from_utf8(text_bytes) {
                                            return Some(text.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        pos += 12 + length;
    }
    None
}

fn parse_parameter_string(s: &str) -> ParsedMetadata {
    let mut prompt = String::new();
    let mut negative_prompt = String::new();
    let mut steps = None;
    let mut guidance = None;
    let mut seed = None;
    let mut width = None;
    let mut height = None;
    let mut model = None;

    let lines: Vec<&str> = s.lines().collect();
    if lines.is_empty() {
        return ParsedMetadata { prompt, negative_prompt, steps, guidance, seed, width, height, model };
    }

    let mut pos = 0;
    let mut prompt_lines = Vec::new();
    while pos < lines.len() && !lines[pos].starts_with("Negative prompt:") && !lines[pos].contains("Steps:") {
        prompt_lines.push(lines[pos]);
        pos += 1;
    }
    prompt = prompt_lines.join("\n").trim().to_string();

    if pos < lines.len() && lines[pos].starts_with("Negative prompt:") {
        let mut neg_lines = Vec::new();
        let first_neg = lines[pos].trim_start_matches("Negative prompt:").trim();
        neg_lines.push(first_neg);
        pos += 1;
        while pos < lines.len() && !lines[pos].contains("Steps:") {
            neg_lines.push(lines[pos]);
            pos += 1;
        }
        negative_prompt = neg_lines.join("\n").trim().to_string();
    }

    if pos < lines.len() {
        let meta_line = lines[pos];
        for field in meta_line.split(',') {
            let field = field.trim();
            if let Some(val) = field.strip_prefix("Steps:") {
                steps = val.trim().parse().ok();
            } else if let Some(val) = field.strip_prefix("CFG scale:") {
                guidance = val.trim().parse().ok();
            } else if let Some(val) = field.strip_prefix("Seed:") {
                seed = val.trim().parse().ok();
            } else if let Some(val) = field.strip_prefix("Size:") {
                let parts: Vec<&str> = val.trim().split('x').collect();
                if parts.len() == 2 {
                    width = parts[0].parse().ok();
                    height = parts[1].parse().ok();
                }
            } else if let Some(val) = field.strip_prefix("Model:") {
                model = Some(val.trim().to_string());
            }
        }
    }

    ParsedMetadata {
        prompt,
        negative_prompt,
        steps,
        guidance,
        seed,
        width,
        height,
        model,
    }
}

struct DbState {
    conn: Mutex<rusqlite::Connection>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DbConversation {
    id: i64,
    title: String,
    created_at: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DbMessage {
    id: i64,
    conversation_id: i64,
    role: String,
    content: String,
    thinking: Option<String>,
    duration_ms: Option<i64>,
    diffusion_steps_json: Option<String>,
}

#[tauri::command]
fn db_get_conversations(state: State<'_, DbState>) -> Result<Vec<DbConversation>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT id, title, created_at FROM conversations ORDER BY id DESC")
        .map_err(|e| e.to_string())?;
    
    let rows = stmt
        .query_map([], |row| {
            Ok(DbConversation {
                id: row.get(0)?,
                title: row.get(1)?,
                created_at: row.get(2)?,
            })
        })
        .map_err(|e| e.to_string())?;

    let mut list = Vec::new();
    for row in rows {
        list.push(row.map_err(|e| e.to_string())?);
    }
    Ok(list)
}

#[tauri::command]
fn db_create_conversation(title: String, state: State<'_, DbState>) -> Result<i64, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO conversations (title) VALUES (?)",
        [&title],
    )
    .map_err(|e| e.to_string())?;
    Ok(conn.last_insert_rowid())
}

#[tauri::command]
fn db_get_messages(conversation_id: i64, state: State<'_, DbState>) -> Result<Vec<DbMessage>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT id, conversation_id, role, content, thinking, duration_ms, diffusion_steps_json FROM messages WHERE conversation_id = ? ORDER BY id ASC")
        .map_err(|e| e.to_string())?;
    
    let rows = stmt
        .query_map([conversation_id], |row| {
            Ok(DbMessage {
                id: row.get(0)?,
                conversation_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                thinking: row.get(4)?,
                duration_ms: row.get(5)?,
                diffusion_steps_json: row.get(6)?,
            })
        })
        .map_err(|e| e.to_string())?;

    let mut list = Vec::new();
    for row in rows {
        list.push(row.map_err(|e| e.to_string())?);
    }
    Ok(list)
}

#[tauri::command]
fn db_add_message(
    conversation_id: i64,
    role: String,
    content: String,
    thinking: Option<String>,
    duration_ms: Option<i64>,
    diffusion_steps_json: Option<String>,
    state: State<'_, DbState>,
) -> Result<i64, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, thinking, duration_ms, diffusion_steps_json) VALUES (?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            conversation_id,
            role,
            content,
            thinking,
            duration_ms,
            diffusion_steps_json
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(conn.last_insert_rowid())
}

#[tauri::command]
fn db_delete_conversation(conversation_id: i64, state: State<'_, DbState>) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "DELETE FROM messages WHERE conversation_id = ?",
        [conversation_id],
    )
    .map_err(|e| e.to_string())?;
    
    conn.execute(
        "DELETE FROM conversations WHERE id = ?",
        [conversation_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn db_update_conversation_title(
    conversation_id: i64,
    title: String,
    state: State<'_, DbState>,
) -> Result<(), String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        rusqlite::params![title, conversation_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DbImage {
    id: i64,
    conversation_id: i64,
    image_base64: String,
    prompt: String,
    negative_prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    seed: u32,
    duration_ms: i64,
    timestamp: i64,
}

#[tauri::command]
fn db_get_images(conversation_id: i64, state: State<'_, DbState>) -> Result<Vec<DbImage>, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT id, conversation_id, image_base64, prompt, negative_prompt, width, height, steps, guidance, seed, duration_ms, timestamp FROM images WHERE conversation_id = ? ORDER BY id DESC")
        .map_err(|e| e.to_string())?;
    
    let rows = stmt
        .query_map([conversation_id], |row| {
            Ok(DbImage {
                id: row.get(0)?,
                conversation_id: row.get(1)?,
                image_base64: row.get(2)?,
                prompt: row.get(3)?,
                negative_prompt: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                steps: row.get(7)?,
                guidance: row.get(8)?,
                seed: row.get(9)?,
                duration_ms: row.get(10)?,
                timestamp: row.get(11)?,
            })
        })
        .map_err(|e| e.to_string())?;

    let mut list = Vec::new();
    for row in rows {
        list.push(row.map_err(|e| e.to_string())?);
    }
    Ok(list)
}

#[tauri::command]
fn db_add_image(
    conversation_id: i64,
    image_base64: String,
    prompt: String,
    negative_prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    seed: u32,
    duration_ms: i64,
    timestamp: i64,
    state: State<'_, DbState>,
) -> Result<i64, String> {
    let conn = state.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO images (conversation_id, image_base64, prompt, negative_prompt, width, height, steps, guidance, seed, duration_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            conversation_id,
            image_base64,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance,
            seed,
            duration_ms,
            timestamp
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(conn.last_insert_rowid())
}

fn clean_api_response(raw: &str) -> String {
    let without_timing = raw
        .lines()
        .take_while(|line| !line.trim_start().starts_with("total time:"))
        .collect::<Vec<_>>()
        .join("\n");

    let split = if let Some(index) = without_timing.rfind("<channel|>") {
        Some((index, "<channel|>".len()))
    } else if let Some(index) = without_timing.rfind("<|channel>final") {
        Some((index, "<|channel>final".len()))
    } else if let Some(index) = without_timing.rfind("<|channel|>final") {
        Some((index, "<|channel|>final".len()))
    } else {
        None
    };

    if let Some((index, marker_length)) = split {
        strip_model_tokens(&without_timing[index + marker_length..])
    } else {
        strip_model_tokens(&without_timing)
    }
}

fn add_cors_headers<R>(response: &mut tiny_http::Response<R>)
where
    R: std::io::Read,
{
    response.add_header(tiny_http::Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
    response.add_header(tiny_http::Header::from_bytes(&b"Access-Control-Allow-Methods"[..], &b"GET, POST, OPTIONS"[..]).unwrap());
    response.add_header(tiny_http::Header::from_bytes(&b"Access-Control-Allow-Headers"[..], &b"Content-Type, Authorization"[..]).unwrap());
}

fn handle_api_request(mut request: tiny_http::Request, app: AppHandle) -> Result<(), String> {
    let url_str = format!("http://localhost:1234{}", request.url());
    let parsed_url = url::Url::parse(&url_str).map_err(|e| e.to_string())?;
    let path = parsed_url.path();
    let method = request.method();

    if method == &tiny_http::Method::Options {
        let mut response = tiny_http::Response::empty(200);
        add_cors_headers(&mut response);
        let _ = request.respond(response);
        return Ok(());
    }

    if method == &tiny_http::Method::Get && path == "/v1/models" {
        let models_json = serde_json::json!({
            "object": "list",
            "data": [
                {
                    "id": "local-model",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "local"
                }
            ]
        });
        let body = models_json.to_string();
        let mut response = tiny_http::Response::from_string(body)
            .with_status_code(200);
        add_cors_headers(&mut response);
        response.add_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
        let _ = request.respond(response);
        return Ok(());
    }

    if method == &tiny_http::Method::Post && path == "/v1/chat/completions" {
        let mut content = String::new();
        request.as_reader().read_to_string(&mut content).map_err(|e| e.to_string())?;

        let req_body: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        
        let messages = req_body.get("messages")
            .and_then(|m| m.as_array())
            .ok_or("Falta campo messages")?;

        let mut prompt = String::new();
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if role == "user" {
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", content));
            } else {
                prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", content));
            }
        }
        prompt.push_str("<start_of_turn>model\n");

        let stream = req_body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

        let runner_state = app.state::<RunnerState>();
        let mut stdin_guard = runner_state.stdin.lock().map_err(|e| e.to_string())?;
        
        if stdin_guard.is_none() {
            let error_json = serde_json::json!({
                "error": {
                    "message": "El modelo no está cargado en VRAM. Por favor, inicia el modelo en la interfaz de Model Studio primero.",
                    "type": "invalid_request_error",
                    "param": null,
                    "code": null
                }
            });
            let body = error_json.to_string();
            let mut response = tiny_http::Response::from_string(body)
                .with_status_code(503);
            add_cors_headers(&mut response);
            response.add_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            let _ = request.respond(response);
            return Ok(());
        }

        let (tx, rx) = std::sync::mpsc::channel::<String>();
        {
            let mut stream_guard = runner_state.active_api_stream.lock().map_err(|e| e.to_string())?;
            *stream_guard = Some(tx);
        }

        if let Some(ref mut stdin) = *stdin_guard {
            let prompt_clean = prompt.replace("\r", "").replace("\n", " ");
            writeln!(stdin, "{}", prompt_clean).map_err(|e| e.to_string())?;
            stdin.flush().map_err(|e| e.to_string())?;
        }

        drop(stdin_guard);

        if stream {
            let mut writer = request.into_writer();
            
            let mut headers_str = String::from("HTTP/1.1 200 OK\r\n");
            headers_str.push_str("Content-Type: text/event-stream\r\n");
            headers_str.push_str("Cache-Control: no-cache\r\n");
            headers_str.push_str("Connection: keep-alive\r\n");
            headers_str.push_str("Access-Control-Allow-Origin: *\r\n");
            headers_str.push_str("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
            headers_str.push_str("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
            headers_str.push_str("\r\n");
            writer.write_all(headers_str.as_bytes()).map_err(|e| e.to_string())?;

            while let Ok(line) = rx.recv() {
                let chunk_json = serde_json::json!({
                    "choices": [
                        {
                            "delta": {
                                "content": line
                            }
                        }
                    ]
                });
                let sse_line = format!("data: {}\n\n", chunk_json.to_string());
                writer.write_all(sse_line.as_bytes()).map_err(|e| e.to_string())?;
                writer.flush().map_err(|e| e.to_string())?;
            }

            writer.write_all(b"data: [DONE]\n\n").map_err(|e| e.to_string())?;
            writer.flush().map_err(|e| e.to_string())?;
        } else {
            let mut full_text = String::new();
            while let Ok(line) = rx.recv() {
                full_text.push_str(&line);
                full_text.push('\n');
            }

            let response_json = serde_json::json!({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": clean_api_response(&full_text)
                        }
                    }
                ]
            });
            let body = response_json.to_string();
            let mut response = tiny_http::Response::from_string(body)
                .with_status_code(200);
            add_cors_headers(&mut response);
            response.add_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            let _ = request.respond(response);
        }
        return Ok(());
    }

    let mut response = tiny_http::Response::from_string("No encontrado")
        .with_status_code(404);
    add_cors_headers(&mut response);
    let _ = request.respond(response);
    Ok(())
}

#[tauri::command]
fn parse_png_metadata(file_path: String) -> Result<ParsedMetadata, String> {
    let bytes = std::fs::read(&file_path)
        .map_err(|e| format!("No se pudo leer el archivo: {e}"))?;
    
    if let Some(params) = read_png_parameters(&bytes) {
        Ok(parse_parameter_string(&params))
    } else {
        Err("No se encontraron metadatos de generación en esta imagen. Asegúrate de que sea un PNG generado por stable-diffusion.cpp o Automatic1111.".to_string())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let app_data = app.path().app_data_dir()?;
            std::fs::create_dir_all(&app_data)?;
            let db_path = app_data.join("local_model_studio.db");
            
            let conn = rusqlite::Connection::open(&db_path)?;
            conn.execute("PRAGMA foreign_keys = ON", [])?;
            
            conn.execute(
                "CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )",
                [],
            )?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    thinking TEXT,
                    duration_ms INTEGER,
                    diffusion_steps_json TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )",
                [],
            )?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    image_base64 TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    steps INTEGER NOT NULL,
                    guidance REAL NOT NULL,
                    seed INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )",
                [],
            )?;

            let app_handle = app.handle().clone();
            std::thread::spawn(move || {
                let server = match tiny_http::Server::http("127.0.0.1:1234") {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("No se pudo iniciar el servidor API: {e}");
                        return;
                    }
                };

                for request in server.incoming_requests() {
                    let app_clone = app_handle.clone();
                    std::thread::spawn(move || {
                        if let Err(e) = handle_api_request(request, app_clone) {
                            eprintln!("Error al procesar petición API: {e}");
                        }
                    });
                }
            });

            app.manage(DbState { conn: Mutex::new(conn) });
            Ok(())
        })
        .manage(RunnerState::default())
        .manage(ImageRunnerState::default())
        .manage(GpuCache::default())
        .invoke_handler(tauri::generate_handler![
            generate,
            stop_generation,
            get_system_status,
            refresh_gpu_detection,
            get_default_paths,
            start_model,
            stop_model,
            send_interactive_prompt,
            is_model_loaded,
            generate_image,
            stop_image_generation,
            is_image_generating,
            get_default_image_paths,
            parse_png_metadata,
            db_get_conversations,
            db_create_conversation,
            db_get_messages,
            db_add_message,
            db_delete_conversation,
            db_update_conversation_title,
            db_get_images,
            db_add_image
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
