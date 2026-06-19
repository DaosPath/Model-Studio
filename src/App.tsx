import {
  ComponentPropsWithoutRef,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

type Role = "user" | "assistant";
type EngineKind = "diffusion" | "llm" | "image";

type SuperAgentStep = {
  id: string;
  name: string;
  status: "idle" | "running" | "completed" | "failed";
  output?: string;
};

type Message = {
  id: number;
  role: Role;
  content: string;
  thinking?: string;
  durationMs?: number;
  pending?: boolean;
  error?: boolean;
  diffusionSteps?: DiffusionProgress[];
  superAgentSteps?: SuperAgentStep[];
};

type Conversation = {
  id: number;
  title: string;
  created_at: string;
  project_id?: number | null;
};

type Project = {
  id: number;
  name: string;
  path: string;
  created_at: string;
};

type CustomTool = {
  id: number;
  name: string;
  description: string;
  command_template: string;
  created_at: string;
};

type DiffusionProgress = {
  v?: number;
  step: number;
  total_steps: number;
  resolved: number;
  total: number;
  text: string;
  tokens?: string[];
  token_types?: string[];
  entropy?: (number | null)[];
  mean_entropy?: number;
  step_ms?: number;
};

type GenerationResult = {
  answer: string;
  thinking?: string;
  duration_ms?: number;
};

type SystemStatus = {
  gpu_name: string;
  gpu_vendor: string;
  gpu_detected: boolean;
  gpu_supports_metrics: boolean;
  vram_used_mb: number;
  vram_total_mb: number;
  gpu_utilization: number;
  temperature: number;
  model_exists: boolean;
  runner_exists: boolean;
};

type RunnerEvent = {
  stream: "stdout" | "stderr" | "system";
  line: string;
};

type ImageProgress = {
  step: number;
  total_steps: number;
  message: string;
  preview_base64?: string;
};

type GeneratedImage = {
  id: number;
  base64: string;
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
  durationMs: number;
  timestamp: number;
};

type ImageResult = {
  image_base64: string;
  output_path: string;
  duration_ms: number;
};

const DEFAULT_RUNNER = "runtime\\llama-diffusion-cli.exe";
const DEFAULT_MODEL = "";
const IS_TAURI = "__TAURI_INTERNALS__" in window;

const initialMessages: Message[] = [
  {
    id: 1,
    role: "assistant",
    content:
      "DiffusionGemma está instalado y listo para generar respuestas localmente.",
  },
];

function LogoIcon() {
  return (
    <svg viewBox="0 0 24 24" className="app-logo-svg" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M5 6.5L12 3L19 6.5V17.5L12 21L5 17.5V6.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
        opacity="0.55"
      />
      <path
        d="M5 6.5L12 10L19 6.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
        opacity="0.4"
      />
      <path
        d="M12 10V21"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
        opacity="0.7"
      />
      <circle cx="12" cy="13" r="2.4" fill="currentColor" />
    </svg>
  );
}

function GpuIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <rect x="2" y="3" width="12" height="10" rx="1.5" />
      <path d="M2 6h12M2 10h12M5 13v1.5M11 13v1.5" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 4.5A1.5 1.5 0 0 1 4.5 3h7A1.5 1.5 0 0 1 13 4.5v5A1.5 1.5 0 0 1 11.5 11H7l-3 2.5V11H4.5A1.5 1.5 0 0 1 3 9.5v-5Z" />
    </svg>
  );
}

function PowerIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 2v6M5 5a4.5 4.5 0 1 0 6 0" />
    </svg>
  );
}

function AttachIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8.5 3.5l-4 4a2.5 2.5 0 0 0 3.5 3.5l5-5a4 4 0 0 0-5.5-5.5l-5.5 5.5" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 2v3M8 11v3M2 8h3M11 8h3M4 4l1.5 1.5M10.5 10.5 12 12M12 4l-1.5 1.5M5.5 10.5 4 12" />
      <circle cx="8" cy="8" r="1.6" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <rect x="4" y="4" width="8" height="8" rx="1.2" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 8l11-5-4 12-2-5-5-2Z" />
    </svg>
  );
}

function TempIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 2.5a1.8 1.8 0 0 0-1.8 1.8V9a3 3 0 1 0 3.6 0V4.3A1.8 1.8 0 0 0 8 2.5Z" />
      <circle cx="8" cy="11.5" r="1.4" />
    </svg>
  );
}

function UsageIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M2 12l3-3 3 2 3-5 3 3" />
    </svg>
  );
}

function WarningIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 2L1.5 13h13L8 2Z" />
      <path d="M8 6.5v3M8 11.5v.5" />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="m6 3.5 4.5 4.5L6 12.5" />
    </svg>
  );
}

function ModelGlyph() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5l1.6 4.6 4.8.3-3.7 3 1.2 4.7L12 13.6l-3.9 2.5 1.2-4.7-3.7-3 4.8-.3L12 3.5Z" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M13.2 5.5A5.5 5.5 0 1 0 13 11" />
      <path d="M13.2 2.8v2.9h-2.9" />
    </svg>
  );
}

function DiffusionIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="4" cy="8" r="1.5" />
      <circle cx="8" cy="4" r="1.5" />
      <circle cx="12" cy="8" r="1.5" />
      <circle cx="8" cy="12" r="1.5" />
      <path d="M5.2 6.8 6.8 5.2M9.2 5.2l1.6 1.6M10.8 9.2l-1.6 1.6M6.8 10.8 5.2 9.2" />
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <rect x="2" y="3" width="12" height="10" rx="1.5" />
      <circle cx="6" cy="6.5" r="1" />
      <path d="m2 11 3.5-3.5L9 11l2.5-2.5L14 11" />
    </svg>
  );
}

function LlmIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 4.5h6M3 8h10M3 11.5h8" />
      <path d="m11 3 2 1.5L11 6" />
    </svg>
  );
}

function CopyIcon({ copied }: { copied: boolean }) {
  return copied ? (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="m3 8.5 3 3 7-7" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <rect x="5.5" y="2.5" width="8" height="9" rx="1.5" />
      <path d="M10.5 11.5v1A1.5 1.5 0 0 1 9 14H3.5A1.5 1.5 0 0 1 2 12.5V6a1.5 1.5 0 0 1 1.5-1.5h2" />
    </svg>
  );
}

function CodeBlock({
  className,
  children,
  ...props
}: ComponentPropsWithoutRef<"code">) {
  const [copied, setCopied] = useState(false);
  const code = String(children).replace(/\n$/, "");
  const language = /language-([\w-]+)/.exec(className || "")?.[1];
  const isBlock = Boolean(language) || code.includes("\n");

  if (!isBlock) {
    return (
      <code className="inline-code" {...props}>
        {children}
      </code>
    );
  }

  async function copyCode() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div className="code-block">
      <div className="code-header">
        <span>{language || "Código"}</span>
        <button type="button" onClick={copyCode} aria-label="Copiar código">
          <CopyIcon copied={copied} />
          {copied ? "Copiado" : "Copiar"}
        </button>
      </div>
      <pre>
        <code className={className} {...props}>
          {code}
        </code>
      </pre>
    </div>
  );
}

function MarkdownMessage({
  content,
  error,
}: {
  content: string;
  error?: boolean;
}) {
  return (
    <div className={`markdown-message${error ? " message-error" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: CodeBlock,
          pre: ({ children }) => <>{children}</>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

const DEVELOPER_SYSTEM = `Eres un Asistente Programador de Élite (Developer Mode). Tu objetivo es escribir código limpio, eficiente y bien estructurado. Siempre explica tus decisiones de diseño y sigue las mejores prácticas de programación.`;

const RESEARCHER_SYSTEM = `Eres un Asistente Investigador de Élite (Researcher Mode). Tu objetivo es analizar información, buscar hechos, resumir temas complejos y proporcionar respuestas precisas y basadas en datos.`;

const FILE_SPECIALIST_SYSTEM = `Eres un Especialista de Archivos (File Specialist Mode) con acceso al sistema de archivos local a través de herramientas especiales.
Puedes inspeccionar el espacio de trabajo usando las siguientes herramientas. Para usarlas, debes escribir exactamente el comando en una línea nueva, sin texto antes ni después en esa misma línea:

TOOL: read_file <ruta_absoluta>
TOOL: list_dir <ruta_absoluta>

Por ejemplo, si necesitas ver el contenido de 'C:\\proyectos\\main.js', escribe exactamente:
TOOL: read_file C:\\proyectos\\main.js

Y espera a recibir la respuesta (OBSERVATION). No inventes el contenido de los archivos. Analiza el contenido real devuelto por la herramienta.`;

function SuperAgentPanel({
  steps,
  messageId,
  expandedPanels,
  onToggle,
}: {
  steps: SuperAgentStep[];
  messageId: number;
  expandedPanels: Record<number, boolean>;
  onToggle: (id: number) => void;
}) {
  const isExpanded = !!expandedPanels[messageId];

  return (
    <div className="super-agent-panel">
      <div className="super-agent-header" onClick={() => onToggle(messageId)}>
        <span>Panel de Súper Agente (Orquestador)</span>
        <span>{isExpanded ? "▲" : "▼"}</span>
      </div>
      {isExpanded && (
        <div className="super-agent-steps">
          {steps.map((step) => {
            return (
              <div key={step.id} className={`super-agent-step ${step.status}`}>
                <div className="step-row">
                  <span className="step-icon">
                    {step.status === "running" && "⏳"}
                    {step.status === "completed" && "✅"}
                    {step.status === "failed" && "❌"}
                    {step.status === "idle" && "⚪"}
                  </span>
                  <span className="step-name">{step.name}</span>
                </div>
                {step.output && (
                  <pre className="step-output">{step.output}</pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Strip model-specific markers from canvas text for display */
function cleanCanvasTokens(raw: string): string[] {
  return raw
    .replace(/<\|channel\|?>(?:thought|final)?/g, "")
    .replace(/<channel\|>/g, " \u2503 ")  // separator between thought / answer
    .replace(/<\|end\|>/g, "")
    .replace(/<\/s>/g, "")
    .replace(/\n/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function addFinalStep(steps: DiffusionProgress[], answerText: string): DiffusionProgress[] {
  if (steps.length === 0) return steps;
  const last = steps[steps.length - 1];
  const finalStep: DiffusionProgress = {
    step: last.total_steps,
    total_steps: last.total_steps,
    resolved: last.total,
    total: last.total,
    text: answerText,
  };
  // Don't duplicate if already at 100%
  if (last.step === last.total_steps) return steps;
  return [...steps, finalStep];
}

function DiffusionCanvas({
  steps,
  live = false,
}: {
  steps: DiffusionProgress[];
  live?: boolean;
}) {
  const [selectedIndex, setSelectedIndex] = useState(Math.max(0, steps.length - 1));

  useEffect(() => {
    if (live) setSelectedIndex(Math.max(0, steps.length - 1));
  }, [live, steps.length]);

  const frame = steps[Math.min(selectedIndex, steps.length - 1)];
  if (!frame) return null;

  const isComplete = frame.step >= frame.total_steps;
  const progress = frame.total_steps
    ? Math.min(100, Math.round((frame.step / frame.total_steps) * 100))
    : 0;
  const resolved = frame.total
    ? Math.min(100, Math.round((frame.resolved / frame.total) * 100))
    : 0;

  const isV2 = Array.isArray(frame.tokens);
  const tokens = isV2 ? frame.tokens! : cleanCanvasTokens(frame.text).slice(0, 80);
  const previousFrame = selectedIndex > 0 ? steps[selectedIndex - 1] : undefined;
  const previousTokens = previousFrame
    ? cleanCanvasTokens(previousFrame.text).slice(0, 80)
    : undefined;

  const hasTokens = tokens.length > 0;

  const hasMeanEntropy = frame.mean_entropy !== undefined && frame.mean_entropy !== null;
  const confidencePct = hasMeanEntropy 
    ? Math.max(0, Math.min(100, Math.round((1 - Math.min(1.0, frame.mean_entropy! / 1.5)) * 100)))
    : 0;
  const confidenceHue = hasMeanEntropy
    ? 140 - Math.min(1.0, frame.mean_entropy! / 1.5) * 130
    : 140;

  return (
    <div className={`diffusion-canvas${live ? " live" : ""}${isComplete ? " complete" : ""}`}>
      <div className="diffusion-heading">
        <div>
          <span className={`diffusion-icon-dot${isComplete ? " done" : ""}`} aria-hidden="true" />
          <strong>
            {isComplete
              ? "Difusi\u00f3n completada"
              : live
                ? "Difusi\u00f3n en curso"
                : "Proceso de difusi\u00f3n"}
          </strong>
          <span className="diffusion-step-label">
            {frame.step}/{frame.total_steps}
          </span>
        </div>
        <span className={`diffusion-percent${isComplete ? " done" : ""}`}>{progress}%</span>
      </div>
      <div className="diffusion-meter" aria-label={`Progreso: ${progress}%`}>
        <span style={{ width: `${progress}%` }} className={isComplete ? "done" : ""} />
      </div>

      {hasMeanEntropy && (
        <div className="diffusion-confidence-container">
          <div className="diffusion-confidence-label">
            <span>Confianza del modelo</span>
            <strong style={{ color: `hsl(${confidenceHue}, 85%, 65%)` }}>{confidencePct}%</strong>
          </div>
          <div className="diffusion-confidence-bar">
            <span
              style={{
                width: `${confidencePct}%`,
                background: `linear-gradient(90deg, hsl(140, 75%, 45%), hsl(${confidenceHue}, 85%, 50%))`
              }}
            />
          </div>
        </div>
      )}

      {frame.entropy && frame.entropy.length > 0 && (
        (() => {
          let certainCount = 0;
          let mediumCount = 0;
          let uncertainCount = 0;
          frame.entropy.forEach((val) => {
            if (val === null || val === undefined) return;
            if (val < 0.15) certainCount++;
            else if (val < 0.60) mediumCount++;
            else uncertainCount++;
          });
          const totalValid = certainCount + mediumCount + uncertainCount;
          const certainPct = totalValid ? Math.round((certainCount / totalValid) * 100) : 0;
          const mediumPct = totalValid ? Math.round((mediumCount / totalValid) * 100) : 0;
          const uncertainPct = totalValid ? Math.max(0, 100 - certainPct - mediumPct) : 0;

          if (totalValid === 0) return null;

          return (
            <div className="entropy-distribution-container">
              <div className="entropy-distribution-label">
                <span>Distribución de Certeza</span>
                <div className="entropy-legend">
                  <span className="legend-item cert-high">Alta: {certainPct}%</span>
                  <span className="legend-item cert-med">Media: {mediumPct}%</span>
                  <span className="legend-item cert-low">Baja: {uncertainPct}%</span>
                </div>
              </div>
              <div className="entropy-distribution-bar">
                {certainPct > 0 && <span className="dist-segment cert-high" style={{ width: `${certainPct}%` }} title={`Alta certeza: ${certainCount} tokens`} />}
                {mediumPct > 0 && <span className="dist-segment cert-med" style={{ width: `${mediumPct}%` }} title={`Media certeza: ${mediumCount} tokens`} />}
                {uncertainPct > 0 && <span className="dist-segment cert-low" style={{ width: `${uncertainPct}%` }} title={`Baja certeza: ${uncertainCount} tokens`} />}
              </div>
            </div>
          );
        })()
      )}

      {hasTokens ? (
        <div className="token-canvas" aria-label="Lienzo de tokens">
          {tokens.map((token, index) => {
            const tokenType = frame.token_types?.[index];
            const entropyVal = frame.entropy?.[index];

            if (token === "┃" || token === "\u2503" || tokenType === "separator") {
              return <span className="separator" key={`sep-${index}`} aria-hidden="true">┃</span>;
            }

            let cls = "";
            let customStyle: React.CSSProperties = {};

            if (tokenType) {
              cls = tokenType; // "stable", "changed", "masked"
            } else {
              cls = [
                token === "\u25a1" ? "masked" : "",
                previousTokens
                  ? previousTokens[index] === token
                    ? "stable"
                    : "changed"
                  : "",
              ]
                .filter(Boolean)
                .join(" ");
            }

            if (entropyVal !== undefined && entropyVal !== null && cls !== "masked" && cls !== "separator") {
              const heat = Math.min(1.0, entropyVal / 1.5);
              const hue = 140 - heat * 130;
              const sat = 60 + heat * 20;
              const light = 20 + heat * 15;
              const alpha = 0.35 + heat * 0.3;
              customStyle = {
                backgroundColor: `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`,
                borderColor: `hsla(${hue}, ${sat}%, ${30 + heat * 20}%, ${alpha + 0.2})`,
                color: `hsla(${hue}, ${sat}%, ${80 + heat * 10}%, 1)`,
              };
            }

            return (
              <span className={cls} key={`${index}-${token}`} style={customStyle}>
                {token}
              </span>
            );
          })}
        </div>
      ) : (
        <div className="canvas-denoising">
          <div className="denoising-wave">
            {Array.from({ length: 12 }).map((_, i) => (
              <span key={i} style={{ animationDelay: `${i * 0.08}s` }} />
            ))}
          </div>
          <p>Resolviendo tokens mediante difusi\u00f3n discreta\u2026</p>
        </div>
      )}
       <div className="diffusion-footer">
        <div className="diffusion-footer-stats">
          <span className="diffusion-stat">
            {isComplete && <span>✓</span>}
            <strong>{frame.resolved}</strong>/{frame.total} estables
            <span className="diffusion-stat-sep">·</span>
            <strong>{resolved}%</strong>
          </span>
          {frame.step_ms ? (
            <span className="diffusion-stat">
              <strong>{frame.step_ms.toFixed(0)}</strong> ms
              <span className="diffusion-stat-sep">·</span>
              <strong>{Math.round(frame.total / (frame.step_ms / 1000))}</strong> t/s
            </span>
          ) : null}
        </div>
        {!live && steps.length > 1 && (
          <label>
            Paso
            <input
              type="range"
              min="0"
              max={steps.length - 1}
              value={selectedIndex}
              onChange={(event) => setSelectedIndex(Number(event.target.value))}
            />
          </label>
        )}
      </div>
    </div>
  );
}

function formatDuration(durationMs?: number) {
  if (!durationMs) return "unos segundos";
  const seconds = durationMs / 1000;
  return seconds < 10 ? `${seconds.toFixed(1)} s` : `${Math.round(seconds)} s`;
}

function ThinkingBlock({
  thinking,
  durationMs,
}: {
  thinking: string;
  durationMs?: number;
}) {
  return (
    <details className="thinking-block">
      <summary>
        <span className="summary-dot" aria-hidden="true" />
        <span>Pensó durante {formatDuration(durationMs)}</span>
        <ChevronIcon />
      </summary>
      <div className="thinking-content">{thinking}</div>
    </details>
  );
}

function WelcomeState({ onSuggestion }: { onSuggestion: (text: string) => void }) {
  const suggestions = [
    "Explica la fotosíntesis en términos simples",
    "Escribe un poema corto sobre la noche",
    "Dame tres ideas para un proyecto local",
    "Resume las ventajas de la inferencia local",
  ];
  return (
    <div className="welcome">
      <div className="welcome-orb" aria-hidden="true">
        <SparkleIcon />
      </div>
      <h2>Listo para razonar en paralelo</h2>
      <p>
        DiffusionGemma 26B A4B ejecuta todo en tu GPU mediante difusión discreta. Sin
        nube, sin telemetría, sin esperas.
      </p>
      <div className="welcome-suggestions">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            className="welcome-suggestion"
            onClick={() => onSuggestion(suggestion)}
          >
            <SparkleIcon />
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}

const IMAGE_PRESETS = [
  { label: "512 × 512", w: 512, h: 512 },
  { label: "768 × 768", w: 768, h: 768 },
  { label: "1024 × 1024", w: 1024, h: 1024 },
  { label: "512 × 768", w: 512, h: 768 },
  { label: "768 × 512", w: 768, h: 512 },
];

function ImageStudio({
  imagePrompt, setImagePrompt,
  negativePrompt, setNegativePrompt,
  imageWidth, setImageWidth,
  imageHeight, setImageHeight,
  imageSteps, setImageSteps,
  imageGuidance, setImageGuidance,
  imageSeed, setImageSeed,
  imageGenerating,
  imageProgress,
  generatedImages,
  selectedImageId, setSelectedImageId,
  onGenerate, onStop,
}: {
  imagePrompt: string;
  setImagePrompt: (v: string) => void;
  negativePrompt: string;
  setNegativePrompt: (v: string) => void;
  imageWidth: number;
  setImageWidth: (v: number) => void;
  imageHeight: number;
  setImageHeight: (v: number) => void;
  imageSteps: number;
  setImageSteps: (v: number) => void;
  imageGuidance: number;
  setImageGuidance: (v: number) => void;
  imageSeed: number;
  setImageSeed: (v: number) => void;
  imageGenerating: boolean;
  imageProgress: ImageProgress | null;
  generatedImages: GeneratedImage[];
  selectedImageId: number | null;
  setSelectedImageId: (v: number | null) => void;
  onGenerate: () => void;
  onStop: () => void;
}) {
  const selectedImage = generatedImages.find((img) => img.id === selectedImageId);
  const progressPercent = imageProgress && imageProgress.total_steps > 0
    ? Math.round((imageProgress.step / imageProgress.total_steps) * 100)
    : 0;

  return (
    <section className="image-studio">
      <div className="image-prompt-card">
        <div className="image-prompt-main">
          <textarea
            value={imagePrompt}
            onChange={(e) => setImagePrompt(e.target.value)}
            placeholder="Describe la imagen que quieres generar…"
            rows={2}
            disabled={imageGenerating}
            aria-label="Prompt de imagen"
          />
          <input
            type="text"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="Prompt negativo (opcional): lo que NO quieres ver"
            disabled={imageGenerating}
            aria-label="Prompt negativo"
            className="negative-prompt-input"
          />
        </div>
        <div className="image-prompt-actions">
          {imageGenerating ? (
            <button type="button" className="button danger" onClick={onStop}>
              <StopIcon />
              Detener
            </button>
          ) : (
            <button
              type="button"
              className="button primary"
              onClick={onGenerate}
              disabled={!imagePrompt.trim()}
            >
              <SparkleIcon />
              Generar
            </button>
          )}
        </div>
      </div>

      <div className="image-settings-row">
        <div className="image-presets">
          {IMAGE_PRESETS.map((preset) => {
              const active = imageWidth === preset.w && imageHeight === preset.h;
              return (
                <button
                  key={preset.label}
                  type="button"
                  className={`image-preset${active ? " active" : ""}`}
                  onClick={() => { setImageWidth(preset.w); setImageHeight(preset.h); }}
                  disabled={imageGenerating}
                >
                  {preset.label}
                </button>
              );
            })}
        </div>
        <div className="image-settings-fields">
          <label>
            <span>Steps</span>
            <input type="number" min="1" max="200" value={imageSteps}
              onChange={(e) => setImageSteps(Number(e.target.value))}
              disabled={imageGenerating} />
          </label>
          <label>
            <span>CFG</span>
            <input type="number" min="0" max="20" step="0.5" value={imageGuidance}
              onChange={(e) => setImageGuidance(Number(e.target.value))}
              disabled={imageGenerating} />
          </label>
          <label>
            <span>Seed</span>
            <input type="number" min="0" value={imageSeed}
              onChange={(e) => setImageSeed(Number(e.target.value))}
              disabled={imageGenerating} placeholder="auto" />
          </label>
        </div>
      </div>

      <div className="image-preview-area">
        {imageGenerating ? (
          <div className="image-preview-generating">
            {imageProgress?.preview_base64 ? (
              <img src={`data:image/png;base64,${imageProgress.preview_base64}`} alt="Denoising..." className="generating-preview-img" />
            ) : (
              <div className="generating-orb" aria-hidden="true">
                <SparkleIcon />
              </div>
            )}
            <div className="generating-info">
              <strong>Generando imagen…</strong>
              {imageProgress && (
                <>
                  <div className="generating-progress">
                    <span style={{ width: `${progressPercent}%` }} />
                  </div>
                  <span className="generating-step">
                    Paso {imageProgress.step}/{imageProgress.total_steps} · {progressPercent}%
                  </span>
                </>
              )}
            </div>
          </div>
        ) : selectedImage ? (
          <div className="image-preview-loaded">
            <img src={`data:image/png;base64,${selectedImage.base64}`} alt={selectedImage.prompt} />
            <div className="image-preview-meta">
              <span>{selectedImage.width}×{selectedImage.height}</span>
              <span>{selectedImage.steps} pasos</span>
              <span>CFG {selectedImage.guidance}</span>
              <span>seed {selectedImage.seed}</span>
              <span>{(selectedImage.durationMs / 1000).toFixed(1)}s</span>
            </div>
          </div>
        ) : (
          <div className="image-preview-empty">
            <div className="empty-orb" aria-hidden="true">
              <ImageIcon />
            </div>
            <h2>Tu lienzo está vacío</h2>
            <p>Escribe un prompt arriba y pulsa <strong>Generar</strong> para crear una imagen localmente.</p>
          </div>
        )}
      </div>

      {generatedImages.length > 0 && (
        <div className="image-history-strip">
          {generatedImages.map((img) => (
            <button
              key={img.id}
              type="button"
              className={`image-thumb${img.id === selectedImageId ? " active" : ""}`}
              onClick={() => setSelectedImageId(img.id)}
              title={img.prompt}
            >
              <img src={`data:image/png;base64,${img.base64}`} alt={img.prompt} />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function resolvePath(userPath: string, projectCwd: string) {
  if (!projectCwd) return userPath;
  const isAbsolute = /^[a-zA-Z]:\\|^[a-zA-Z]:\/|^\/|^\\\\/.test(userPath);
  if (isAbsolute) return userPath;
  const separator = projectCwd.endsWith("\\") || projectCwd.endsWith("/") ? "" : "\\";
  return projectCwd + separator + userPath;
}

function parseModelResponseJS(raw: string) {
  const clean = (val: string) => val
    .replace(/<\|channel>thought/g, "")
    .replace(/<\|channel\|>thought/g, "")
    .replace(/<\|channel>final/g, "")
    .replace(/<\|channel\|>final/g, "")
    .replace(/<channel\|>/g, "")
    .replace(/<\|end\|>/g, "")
    .replace(/<\/s>/g, "")
    .trim();

  let splitIndex = -1;
  let markerLen = 0;

  if (raw.includes("<channel|>")) {
    splitIndex = raw.lastIndexOf("<channel|>");
    markerLen = "<channel|>".length;
  } else if (raw.includes("<|channel>final")) {
    splitIndex = raw.lastIndexOf("<|channel>final");
    markerLen = "<|channel>final".length;
  } else if (raw.includes("<|channel|>final")) {
    splitIndex = raw.lastIndexOf("<|channel|>final");
    markerLen = "<|channel|>final".length;
  }

  if (splitIndex !== -1) {
    const thought = clean(raw.substring(0, splitIndex));
    const answer = clean(raw.substring(splitIndex + markerLen));
    return {
      thinking: thought || undefined,
      answer: answer
    };
  } else {
    return {
      thinking: undefined,
      answer: clean(raw)
    };
  }
}

function App() {
  const [engineKind, setEngineKind] = useState<EngineKind>("diffusion");
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [chatMode, setChatMode] = useState<"chat" | "agent" | "super-agent">("chat");
  const [selectedAgent, setSelectedAgent] = useState<"developer" | "researcher" | "file-specialist">("developer");
  const [expandedPanels, setExpandedPanels] = useState<Record<number, boolean>>({});
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<number | null>(null);
  const [customTools, setCustomTools] = useState<CustomTool[]>([]);
  const [showProjectModal, setShowProjectModal] = useState(false);
  const [showToolModal, setShowToolModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectPath, setNewProjectPath] = useState("");
  const [newToolName, setNewToolName] = useState("");
  const [newToolDescription, setNewToolDescription] = useState("");
  const [newToolCommandTemplate, setNewToolCommandTemplate] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [editingConversationId, setEditingConversationId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const messagesRef = useRef<Message[]>([]);
  messagesRef.current = messages;

  const [prompt, setPrompt] = useState("");
  const [running, setRunning] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [runnerPath, setRunnerPath] = useState(DEFAULT_RUNNER);
  const [modelStatus, setModelStatus] = useState<"stopped" | "loading" | "ready">("stopped");
  const activeInteractiveMessageId = useRef<number | null>(null);
  const [modelPath, setModelPath] = useState(DEFAULT_MODEL);
  const [gpuLayers, setGpuLayers] = useState(20);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [cfgScale, setCfgScale] = useState(0.0);
  const [stability, setStability] = useState(1);
  const [tMin, setTMin] = useState(0.4);
  const [tMax, setTMax] = useState(0.8);
  const [entropyBound, setEntropyBound] = useState(0.1);
  const [confidence, setConfidence] = useState(0.005);
  const [logs, setLogs] = useState<string[]>([
    "[sistema] Aplicación iniciada",
    "[sistema] Esperando verificación del runner",
  ]);
  const [status, setStatus] = useState<SystemStatus>({
    gpu_name: "Detectando…",
    gpu_vendor: "",
    gpu_detected: false,
    gpu_supports_metrics: false,
    vram_used_mb: 0,
    vram_total_mb: 0,
    gpu_utilization: 0,
    temperature: 0,
    model_exists: false,
    runner_exists: false,
  });
  const activeAssistantId = useRef<number | null>(null);
  const activeSuperAgentStepId = useRef<string | null>(null);
  const pendingToolCall = useRef<{ type: string; argument: string } | null>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const projectsRef = useRef<Project[]>([]);
  projectsRef.current = projects;
  
  const activeProjectIdRef = useRef<number | null>(null);
  activeProjectIdRef.current = activeProjectId;
  
  const customToolsRef = useRef<CustomTool[]>([]);
  customToolsRef.current = customTools;

  async function loadProjects() {
    try {
      const list = await invoke<Project[]>("db_get_projects");
      setProjects(list);
    } catch (e) {
      console.error("Error loading projects:", e);
    }
  }

  async function loadCustomTools() {
    try {
      const list = await invoke<CustomTool[]>("db_get_custom_tools");
      setCustomTools(list);
    } catch (e) {
      console.error("Error loading custom tools:", e);
    }
  }

  async function handleCreateProject(e: FormEvent) {
    e.preventDefault();
    const name = newProjectName.trim();
    const path = newProjectPath.trim();
    if (!name || !path) return;
    try {
      await invoke("db_create_project", { name, path });
      setNewProjectName("");
      setNewProjectPath("");
      setShowProjectModal(false);
      await loadProjects();
    } catch (err) {
      alert(`Error al crear proyecto: ${err}`);
    }
  }

  async function deleteProject(id: number) {
    if (!window.confirm("¿Seguro de que deseas eliminar este proyecto? Los chats vinculados también se perderán.")) return;
    try {
      await invoke("db_delete_project", { id });
      if (activeProjectId === id) {
        setActiveProjectId(null);
      }
      await loadProjects();
    } catch (err) {
      alert(`Error al eliminar proyecto: ${err}`);
    }
  }

  async function handleCreateTool(e: FormEvent) {
    e.preventDefault();
    const name = newToolName.trim().toLowerCase().replace(/[^a-z0-9_-]/g, "");
    const desc = newToolDescription.trim();
    const template = newToolCommandTemplate.trim();
    if (!name || !desc || !template) return;
    try {
      await invoke("db_create_custom_tool", { name, description: desc, commandTemplate: template });
      setNewToolName("");
      setNewToolDescription("");
      setNewToolCommandTemplate("");
      await loadCustomTools();
    } catch (err) {
      alert(`Error al crear herramienta: ${err}`);
    }
  }

  async function handleDeleteTool(id: number) {
    if (!window.confirm("¿Seguro que deseas eliminar esta herramienta?")) return;
    try {
      await invoke("db_delete_custom_tool", { id });
      await loadCustomTools();
    } catch (err) {
      alert(`Error al eliminar herramienta: ${err}`);
    }
  }

  const [imagePrompt, setImagePrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [imageWidth, setImageWidth] = useState(512);
  const [imageHeight, setImageHeight] = useState(512);
  const [imageSteps, setImageSteps] = useState(30);
  const [imageGuidance, setImageGuidance] = useState(7.0);
  const [imageSeed, setImageSeed] = useState(0);
  const [imageRunnerPath, setImageRunnerPath] = useState("runtime\\sd-cli.exe");
  const [imageModelPath, setImageModelPath] = useState("");
  const [imageLoraDir, setImageLoraDir] = useState("");
  const [imageGenerating, setImageGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([]);
  const [imageProgress, setImageProgress] = useState<ImageProgress | null>(null);
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const imageNextId = useRef(1);

  const refreshStatus = async () => {
    if (!IS_TAURI) return;
    try {
      await invoke("refresh_gpu_detection").catch(() => {});
      const next = await invoke<SystemStatus>("get_system_status", {
        runnerPath,
        modelPath,
      });
      setStatus(next);
    } catch (error) {
      setLogs((current) => [...current.slice(-199), `[error] ${String(error)}`]);
    }
  };

  useEffect(() => {
    if (!IS_TAURI) return;
    invoke<{ runner_path: string; model_path: string }>("get_default_paths")
      .then((paths) => {
        setRunnerPath(paths.runner_path);
        setModelPath(paths.model_path);
      })
      .catch(() => {
        // Development fallback paths are already initialized.
      });

    invoke<boolean>("is_model_loaded").then((loaded) => {
      setModelStatus(loaded ? "ready" : "stopped");
    });

    loadProjects();
    loadCustomTools();
  }, []);

  useEffect(() => {
    if (!IS_TAURI) return;
    loadConversations(activeProjectId);
  }, [activeProjectId]);

  async function loadConversations(projectId: number | null = activeProjectId) {
    try {
      const list = await invoke<Conversation[]>("db_get_conversations", { projectId });
      setConversations(list);
      if (list.length > 0) {
        await selectConversation(list[0].id);
      } else {
        await createNewConversation("Prueba local", projectId);
      }
    } catch (error) {
      console.error("Error al cargar conversaciones:", error);
    }
  }

  async function selectConversation(id: number) {
    if (running) return;
    setActiveConversationId(id);
    try {
      const dbMsgs = await invoke<any[]>("db_get_messages", { conversationId: id });
      const mappedMsgs: Message[] = dbMsgs.map((m) => {
        let diffusionSteps: DiffusionProgress[] = [];
        let superAgentSteps: SuperAgentStep[] = [];
        if (m.diffusion_steps_json) {
          try {
            const parsed = JSON.parse(m.diffusion_steps_json);
            if (Array.isArray(parsed) && parsed.length > 0) {
              if (parsed[0] && typeof parsed[0] === "object" && "id" in parsed[0]) {
                superAgentSteps = parsed as SuperAgentStep[];
              } else {
                diffusionSteps = parsed as DiffusionProgress[];
              }
            }
          } catch (e) {
            console.error("Error parsing diffusion steps JSON:", e);
          }
        }
        return {
          id: m.id,
          role: m.role as Role,
          content: m.content,
          thinking: m.thinking || undefined,
          durationMs: m.duration_ms || undefined,
          diffusionSteps: diffusionSteps.length > 0 ? diffusionSteps : undefined,
          superAgentSteps: superAgentSteps.length > 0 ? superAgentSteps : undefined,
        };
      });
      
      if (mappedMsgs.length === 0) {
        setMessages(initialMessages);
      } else {
        setMessages(mappedMsgs);
      }

      // Cargar imágenes generadas para esta conversación
      const dbImgs = await invoke<any[]>("db_get_images", { conversationId: id });
      const mappedImgs: GeneratedImage[] = dbImgs.map((img) => ({
        id: img.id,
        base64: img.image_base64,
        prompt: img.prompt,
        negativePrompt: img.negative_prompt,
        width: img.width,
        height: img.height,
        steps: img.steps,
        guidance: img.guidance,
        seed: img.seed,
        durationMs: img.duration_ms,
        timestamp: img.timestamp,
      }));
      setGeneratedImages(mappedImgs);
      if (mappedImgs.length > 0) {
        setSelectedImageId(mappedImgs[0].id);
      } else {
        setSelectedImageId(null);
      }

    } catch (error) {
      console.error("Error al cargar mensajes o imágenes:", error);
    }
  }

  async function createNewConversation(title: string = "Nueva conversación", projectId: number | null = activeProjectId) {
    if (running) return;
    try {
      const newId = await invoke<number>("db_create_conversation", { title, projectId });
      const list = await invoke<Conversation[]>("db_get_conversations", { projectId });
      setConversations(list);
      setActiveConversationId(newId);
      setMessages(initialMessages);
      setGeneratedImages([]);
      setSelectedImageId(null);
      
      for (const msg of initialMessages) {
        await invoke("db_add_message", {
          conversationId: newId,
          role: msg.role,
          content: msg.content,
          thinking: null,
          durationMs: null,
          diffusionStepsJson: null,
        });
      }
      
      await selectConversation(newId);
      window.setTimeout(() => textareaRef.current?.focus(), 0);
    } catch (error) {
      console.error("Error al crear conversación:", error);
    }
  }

  async function deleteConversation(id: number, event: React.MouseEvent) {
    event.stopPropagation();
    if (running) return;
    if (!window.confirm("¿Estás seguro de que deseas eliminar esta conversación y todos sus mensajes?")) {
      return;
    }
    try {
      await invoke("db_delete_conversation", { conversationId: id });
      const list = await invoke<Conversation[]>("db_get_conversations", { projectId: activeProjectId });
      setConversations(list);
      if (activeConversationId === id) {
        if (list.length > 0) {
          await selectConversation(list[0].id);
        } else {
          await createNewConversation("Prueba local", activeProjectId);
        }
      }
    } catch (error) {
      console.error("Error al eliminar conversación:", error);
    }
  }

  function startEditingConversation(id: number, currentTitle: string, event: React.MouseEvent) {
    event.stopPropagation();
    setEditingConversationId(id);
    setEditTitle(currentTitle);
  }

  async function saveConversationTitle(id: number) {
    const cleanTitle = editTitle.trim();
    if (!cleanTitle) return;
    try {
      await invoke("db_update_conversation_title", { conversationId: id, title: cleanTitle });
      setEditingConversationId(null);
      const list = await invoke<Conversation[]>("db_get_conversations", { projectId: activeProjectId });
      setConversations(list);
    } catch (error) {
      console.error("Error al renombrar conversación:", error);
    }
  }

  function handleRenameKeyDown(event: KeyboardEvent<HTMLInputElement>, id: number) {
    if (event.key === "Enter") {
      saveConversationTitle(id);
    } else if (event.key === "Escape") {
      setEditingConversationId(null);
    }
  }

  useEffect(() => {
    const unlisten = listen<DiffusionProgress>("diffusion-progress", ({ payload }) => {
      const messageId = activeAssistantId.current || activeInteractiveMessageId.current;
      if (messageId === null) return;

      setMessages((current) =>
        current.map((message) => {
          if (message.id !== messageId) return message;
          const steps = message.diffusionSteps ?? [];
          const lastStep = steps.length > 0 ? steps[steps.length - 1] : undefined;
          const nextSteps =
            lastStep?.step === payload.step
              ? [...steps.slice(0, -1), payload]
              : [...steps, payload].slice(-48);
          return { ...message, diffusionSteps: nextSteps };
        }),
      );
    });
    return () => {
      unlisten.then((dispose) => dispose());
    };
  }, []);

  useEffect(() => {
    refreshStatus();
    const timer = window.setInterval(refreshStatus, running ? 1500 : 5000);
    return () => window.clearInterval(timer);
  }, [runnerPath, modelPath, running]);

  useEffect(() => {
    if (!running) {
      setElapsedSeconds(0);
      return;
    }
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    const unlisten = listen<RunnerEvent>("runner-output", ({ payload }) => {
      const line = payload.line.trimEnd();
      if (!line) return;
      setLogs((current) => [
        ...current.slice(-199),
        `[${payload.stream}] ${line}`,
      ]);

      // Alimentar stdout al canvas y contenido para AMBOS modos
      if (payload.stream === "stdout") {
        const msgId = activeInteractiveMessageId.current ?? activeAssistantId.current;
        if (msgId !== null) {
          const trimmed = payload.line.trim();
          const matchReadFile = trimmed.match(/^TOOL:\s*read_file\s+(.+)$/i);
          const matchListDir = trimmed.match(/^TOOL:\s*list_dir\s+(.+)$/i);
          const matchRunCommand = trimmed.match(/^TOOL:\s*run_command\s+(.+)$/i);
          if (matchReadFile) {
            pendingToolCall.current = { type: "read_file", argument: matchReadFile[1].trim() };
          } else if (matchListDir) {
            pendingToolCall.current = { type: "list_dir", argument: matchListDir[1].trim() };
          } else if (matchRunCommand) {
            pendingToolCall.current = { type: "run_command", argument: matchRunCommand[1].trim() };
          } else {
            const matchCustom = trimmed.match(/^TOOL:\s*([a-zA-Z0-9_-]+)(?:\s+(.+))?$/i);
            if (matchCustom) {
              const toolName = matchCustom[1].trim().toLowerCase();
              const foundTool = customToolsRef.current.find(t => t.name.toLowerCase() === toolName);
              if (foundTool) {
                pendingToolCall.current = { type: foundTool.name, argument: (matchCustom[2] || "").trim() };
              }
            }
          }

          if (activeSuperAgentStepId.current) {
            const stepId = activeSuperAgentStepId.current;
            setMessages((current) =>
              current.map((msg) => {
                if (msg.id !== msgId) return msg;
                const steps = msg.superAgentSteps ?? [];
                const nextSteps = steps.map((s) => {
                  if (s.id !== stepId) return s;
                  const newOutput = s.output ? s.output + "\n" + payload.line : payload.line;
                  return { ...s, output: newOutput };
                });
                let newContent = msg.content;
                if (stepId === "synthesizer") {
                  newContent = msg.content ? msg.content + "\n" + payload.line : payload.line;
                }
                return { ...msg, superAgentSteps: nextSteps, content: newContent };
              })
            );
          } else {
            setMessages((current) =>
              current.map((msg) => {
                if (msg.id !== msgId) return msg;
                const newContent = msg.content ? msg.content + "\n" + payload.line : payload.line;

                // Actualizar el último paso de difusión con el contenido acumulado
                const steps = msg.diffusionSteps ?? [];
                const lastStep = steps[steps.length - 1];
                const nextSteps = lastStep
                  ? [
                      ...steps.slice(0, -1),
                      { ...lastStep, text: newContent },
                    ]
                  : steps;

                return { ...msg, content: newContent, diffusionSteps: nextSteps };
              })
            );
          }
        }
      }
    });
    return () => {
      unlisten.then((dispose) => dispose());
    };
  }, []);

  async function finishAndSaveMessage(msgId: number, target: Message) {
    const raw = target.content;
    const parsed = parseModelResponseJS(raw);
    const answer = parsed.answer.trim() || "La generación terminó sin texto.";
    const thinking = parsed.thinking?.trim() || null;
    const finalSteps = addFinalStep(target.diffusionSteps ?? [], raw);
    const stepsJson = finalSteps.length > 0 ? JSON.stringify(finalSteps) : null;
    
    try {
      const dbId = await invoke<number>("db_add_message", {
        conversationId: activeConversationId,
        role: "assistant",
        content: answer,
        thinking: thinking,
        durationMs: null,
        diffusionStepsJson: stepsJson,
      });

      setMessages((current) =>
        current.map((msg) => {
          if (msg.id !== msgId) return msg;
          return {
            ...msg,
            id: dbId,
            content: answer,
            thinking: thinking || undefined,
            pending: false,
            diffusionSteps: finalSteps,
          };
        })
      );
    } catch (error) {
      console.error("Error al guardar mensaje en DB:", error);
      setMessages((current) =>
        current.map((msg) => {
          if (msg.id !== msgId) return msg;
          return {
            ...msg,
            pending: false,
            error: true,
          };
        })
      );
    }
    activeInteractiveMessageId.current = null;
    setRunning(false);
  }

  useEffect(() => {
    if (!IS_TAURI) return;
    const unlisten = listen("generation-finished", async () => {
      if (activeSuperAgentStepId.current) {
        // Super agent handles step transitions locally/sequentially
        return;
      }

      const msgId = activeInteractiveMessageId.current;
      if (msgId === null) return;

      const target = messagesRef.current.find((msg) => msg.id === msgId);
      if (!target) return;

      if (chatMode === "agent" && pendingToolCall.current) {
        const tool = pendingToolCall.current;
        pendingToolCall.current = null; // reset

        const statusMsg = `\n\n⚙️ [Ejecutando herramienta local: ${tool.type} ${tool.argument}...]`;
        setMessages((current) =>
          current.map((msg) => {
            if (msg.id !== msgId) return msg;
            return {
              ...msg,
              content: msg.content + statusMsg,
            };
          })
        );

        const activeProj = projectsRef.current.find(p => p.id === activeProjectIdRef.current);
        const cwd = activeProj ? activeProj.path : "";

        let resultStr = "";
        try {
          if (tool.type === "read_file") {
            const resolvedPath = resolvePath(tool.argument, cwd);
            const content = await invoke<string>("read_local_file", { path: resolvedPath });
            resultStr = `\nOBSERVATION (Contenido de ${tool.argument}):\n\`\`\`\n${content}\n\`\`\``;
          } else if (tool.type === "list_dir") {
            const resolvedPath = resolvePath(tool.argument, cwd);
            const list = await invoke<string[]>("list_local_directory", { path: resolvedPath });
            resultStr = `\nOBSERVATION (Archivos en ${tool.argument}):\n${list.join("\n")}`;
          } else if (tool.type === "run_command") {
            const commandCwd = cwd || ".";
            const output = await invoke<string>("run_local_command", { command: tool.argument, cwd: commandCwd });
            resultStr = `\nOBSERVATION (Resultado del comando):\n\`\`\`\n${output}\n\`\`\``;
          } else {
            const foundTool = customToolsRef.current.find(t => t.name.toLowerCase() === tool.type.toLowerCase());
            if (foundTool) {
              let cmd = foundTool.command_template;
              if (cmd.includes("{{") && cmd.includes("}}")) {
                cmd = cmd.replace(/\{\{[^}]+\}\}/g, tool.argument);
              } else if (tool.argument) {
                cmd = cmd + " " + tool.argument;
              }
              const commandCwd = cwd || ".";
              const output = await invoke<string>("run_local_command", { command: cmd, cwd: commandCwd });
              resultStr = `\nOBSERVATION (Herramienta "${tool.type}" ejecutada):\n\`\`\`\n${output}\n\`\`\``;
            } else {
              resultStr = `\nOBSERVATION (Error): Herramienta "${tool.type}" no encontrada.`;
            }
          }
        } catch (error) {
          resultStr = `\nOBSERVATION (Error al ejecutar herramienta): ${error}`;
        }

        setMessages((current) =>
          current.map((msg) => {
            if (msg.id !== msgId) return msg;
            return {
              ...msg,
              content: msg.content + resultStr,
            };
          })
        );

        try {
          await invoke("send_interactive_prompt", { prompt: resultStr });
        } catch (err) {
          console.error("Error al reenviar prompt de herramienta:", err);
          finishAndSaveMessage(msgId, target);
        }
        return;
      }

      finishAndSaveMessage(msgId, target);
    });

    const statusUnlisten = listen<string>("model-status-changed", ({ payload }) => {
      setModelStatus(payload as "stopped" | "loading" | "ready");
    });

    return () => {
      unlisten.then((dispose) => dispose());
      statusUnlisten.then((dispose) => dispose());
    };
  }, []);

  useEffect(() => {
    conversationRef.current?.scrollTo({
      top: conversationRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, elapsedSeconds]);

  useEffect(() => {
    if (!IS_TAURI) return;
    invoke<{ runner_path: string; model_path: string; lora_dir: string }>("get_default_image_paths")
      .then((paths) => {
        setImageRunnerPath(paths.runner_path);
        setImageModelPath(paths.model_path);
        setImageLoraDir(paths.lora_dir);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!IS_TAURI) return;
    const unlisten = listen<ImageProgress>("image-progress", ({ payload }) => {
      setImageProgress(payload);
    });
    return () => { unlisten.then((d) => d()); };
  }, []);

  useEffect(() => {
    if (!IS_TAURI) return;
    const unlisten = listen<{ paths: string[] }>("tauri://drag-drop", ({ payload }) => {
      if (payload.paths && payload.paths.length > 0) {
        const path = payload.paths[0];
        if (path.toLowerCase().endsWith(".png")) {
          invoke<any>("parse_png_metadata", { filePath: path })
            .then((meta) => {
              setLogs((curr) => [...curr, `[sistema] Parámetros cargados desde la imagen: ${path.split('\\').pop()}`]);
              if (meta.prompt) setImagePrompt(meta.prompt);
              if (meta.negative_prompt) setNegativePrompt(meta.negative_prompt);
              if (meta.steps) setImageSteps(meta.steps);
              if (meta.guidance) setImageGuidance(meta.guidance);
              if (meta.seed) setImageSeed(meta.seed);
              if (meta.width) setImageWidth(meta.width);
              if (meta.height) setImageHeight(meta.height);
            })
            .catch((err) => {
              setLogs((curr) => [...curr, `[error] No se pudo leer la imagen: ${err}`]);
            });
        }
      }
    });
    return () => { unlisten.then((d) => d()); };
  }, []);

  useEffect(() => {
    if (!IS_TAURI) return;
    const unlisten = listen("image-generation-finished", () => {
      setImageProgress(null);
    });
    return () => { unlisten.then((d) => d()); };
  }, []);

  const readiness = useMemo(() => {
    if (!status.gpu_detected) return "Sin GPU";
    if (!status.runner_exists) return "Runner no disponible";
    if (!status.model_exists) return "Modelo no disponible";
    return running ? "Generando" : "Listo";
  }, [status, running]);

  const statusKind = running
    ? "busy"
    : !status.gpu_detected
      ? "error"
      : status.runner_exists && status.model_exists
        ? "ready"
        : "error";

  async function toggleModelPersistent() {
    if (running) return;
    if (modelStatus === "ready") {
      setModelStatus("loading");
      try {
        await invoke("stop_model");
        setModelStatus("stopped");
      } catch (error) {
        setLogs((current) => [...current, `[error] ${String(error)}`]);
        setModelStatus("ready");
      }
    } else if (modelStatus === "stopped") {
      if (!status.gpu_detected) {
        setLogs((current) => [
          ...current,
          "[error] No se puede cargar el modelo: GPU no detectada",
        ]);
        return;
      }
      setModelStatus("loading");
      setLogs((current) => [...current, `[sistema] Cargando modelo en VRAM de forma persistente...`]);
      try {
        await invoke("start_model", {
          runnerPath,
          modelPath,
          gpuLayers,
          maxTokens,
          cfgScale: engineKind === "diffusion" ? (cfgScale || null) : null,
          tMin: engineKind === "diffusion" ? (tMin || null) : null,
          tMax: engineKind === "diffusion" ? (tMax || null) : null,
          entropyBound: engineKind === "diffusion" ? (entropyBound || null) : null,
          stability: engineKind === "diffusion" ? (stability || null) : null,
          confidence: engineKind === "diffusion" ? (confidence || null) : null,
        });
        setModelStatus("ready");
      } catch (error) {
        setLogs((current) => [...current, `[error] No se pudo cargar el modelo: ${String(error)}`]);
        setModelStatus("stopped");
      }
    }
  }

  async function sendPrompt(event?: FormEvent) {
    event?.preventDefault();
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || running || !status.gpu_detected || !status.runner_exists || !status.model_exists) {
      return;
    }
    if (activeConversationId === null) {
      return;
    }

    // Super Agent Mode Pipeline
    if (chatMode === "super-agent") {
      try {
        const userDbId = await invoke<number>("db_add_message", {
          conversationId: activeConversationId,
          role: "user",
          content: cleanPrompt,
          thinking: null,
          durationMs: null,
          diffusionStepsJson: null,
        });

        const assistantId = userDbId + 1;
        const initialSteps: SuperAgentStep[] = [
          { id: "analyzer", name: "1. Analista (Analyzer)", status: "idle" },
          { id: "critic", name: "2. Crítico (Critic)", status: "idle" },
          { id: "researcher", name: "3. Investigador (Researcher)", status: "idle" },
          { id: "validator", name: "4. Validador (Validator)", status: "idle" },
          { id: "synthesizer", name: "5. Sintetizador (Synthesizer)", status: "idle" },
        ];

        setMessages((current) => [
          ...current,
          { id: userDbId, role: "user", content: cleanPrompt },
          {
            id: assistantId,
            role: "assistant",
            content: "",
            pending: true,
            superAgentSteps: initialSteps,
          },
        ]);
        setPrompt("");
        setRunning(true);
        setExpandedPanels((current) => ({ ...current, [assistantId]: true }));

        // Ensure model is loaded persistently
        if (modelStatus !== "ready") {
          setLogs((current) => [
            ...current.slice(-199),
            `[sistema] El modo Súper Agente requiere el modelo en memoria. Iniciando...`,
          ]);
          await invoke("start_model", {
            runnerPath,
            modelPath,
            gpuLayers,
            maxTokens,
            cfgScale: engineKind === "diffusion" ? (cfgScale || null) : null,
            tMin: engineKind === "diffusion" ? (tMin || null) : null,
            tMax: engineKind === "diffusion" ? (tMax || null) : null,
            entropyBound: engineKind === "diffusion" ? (entropyBound || null) : null,
            stability: engineKind === "diffusion" ? (stability || null) : null,
            confidence: engineKind === "diffusion" ? (confidence || null) : null,
          });
          setModelStatus("ready");
        }

        activeInteractiveMessageId.current = assistantId;

        const stepsToRun = [
          {
            id: "analyzer",
            name: "1. Analista (Analyzer)",
            prompt: `[ROL: Analista] Analiza el problema del usuario: "${cleanPrompt}". Propón 3 hipótesis o enfoques conceptuales de resolución con sus porcentajes de viabilidad estimados.`
          },
          {
            id: "critic",
            name: "2. Crítico (Critic)",
            prompt: `[ROL: Crítico] Evalúa los 3 enfoques propuestos por el Analista. Identifica posibles bugs, casos límite, problemas de rendimiento y riesgos de seguridad.`
          },
          {
            id: "researcher",
            name: "3. Investigador (Researcher)",
            prompt: `[ROL: Investigador] Aporta información teórica, mejores prácticas de la industria, o referencias relevantes sobre la solución discutida hasta ahora.`
          },
          {
            id: "validator",
            name: "4. Validador (Validator)",
            prompt: `[ROL: Validador] Verifica la lógica del código o del razonamiento final a la luz de las críticas e investigaciones previas.`
          },
          {
            id: "synthesizer",
            name: "5. Sintetizador (Synthesizer)",
            prompt: `[ROL: Sintetizador] Consolida todo el proceso anterior en una respuesta final pulida, clara y premium para el usuario. Enfócate en dar una solución directa y lista para usar.`
          }
        ];

        for (const step of stepsToRun) {
          activeSuperAgentStepId.current = step.id;
          setMessages((current) =>
            current.map((msg) => {
              if (msg.id !== assistantId) return msg;
              const nextSteps = (msg.superAgentSteps ?? []).map((s) =>
                s.id === step.id ? { ...s, status: "running" as const } : s
              );
              return { ...msg, superAgentSteps: nextSteps };
            })
          );

          const finishedPromise = new Promise<void>((resolve) => {
            const unlisten = listen("generation-finished", () => {
              unlisten.then((dispose) => dispose());
              resolve();
            });
          });

          await invoke("send_interactive_prompt", { prompt: step.prompt });
          await finishedPromise;

          setMessages((current) =>
            current.map((msg) => {
              if (msg.id !== assistantId) return msg;
              const nextSteps = (msg.superAgentSteps ?? []).map((s) =>
                s.id === step.id ? { ...s, status: "completed" as const } : s
              );
              return { ...msg, superAgentSteps: nextSteps };
            })
          );
        }

        activeSuperAgentStepId.current = null;
        const targetMsg = messagesRef.current.find((msg) => msg.id === assistantId);
        if (targetMsg) {
          const finalSteps = targetMsg.superAgentSteps ?? [];
          const dbId = await invoke<number>("db_add_message", {
            conversationId: activeConversationId,
            role: "assistant",
            content: targetMsg.content,
            thinking: null,
            durationMs: null,
            diffusionStepsJson: JSON.stringify(finalSteps),
          });

          setMessages((current) =>
            current.map((msg) => {
              if (msg.id !== assistantId) return msg;
              return {
                ...msg,
                id: dbId,
                pending: false,
              };
            })
          );
        }

      } catch (err) {
        console.error("Error in super-agent pipeline:", err);
        setMessages((current) =>
          current.map((msg) => {
            if (msg.role !== "assistant" || !msg.pending) return msg;
            const nextSteps = (msg.superAgentSteps ?? []).map((s) =>
              s.status === "running" || s.status === "idle" ? { ...s, status: "failed" as const } : s
            );
            return {
              ...msg,
              superAgentSteps: nextSteps,
              pending: false,
              error: true,
              content: msg.content + `\n\n[Error en orquestación multiagente: ${err}]`,
            };
          })
        );
      } finally {
        activeSuperAgentStepId.current = null;
        activeInteractiveMessageId.current = null;
        setRunning(false);
      }
      return;
    }

    // Normal or Agent Mode
    try {
      let promptToSend = cleanPrompt;
      if (chatMode === "agent") {
        const isFirstMessage = messages.filter((m) => m.role === "user").length === 0;
        if (isFirstMessage) {
          let systemPrompt = "";
          if (selectedAgent === "developer") systemPrompt = DEVELOPER_SYSTEM;
          else if (selectedAgent === "researcher") systemPrompt = RESEARCHER_SYSTEM;
          else if (selectedAgent === "file-specialist") systemPrompt = FILE_SPECIALIST_SYSTEM;
          
          if (customTools.length > 0) {
            systemPrompt += `\n\nTambién tienes acceso a las siguientes herramientas personalizadas creadas por el usuario:\n`;
            customTools.forEach((tool) => {
              systemPrompt += `- TOOL: ${tool.name} <argumentos> (${tool.description})\n`;
            });
            systemPrompt += `\nPara llamar a una herramienta personalizada, escribe: TOOL: <nombre> <argumentos> en una nueva línea y espera la respuesta (OBSERVATION).`;
          }

          promptToSend = `${systemPrompt}\n\n[Mensaje del usuario]:\n${cleanPrompt}`;
        }
      }

      const userDbId = await invoke<number>("db_add_message", {
        conversationId: activeConversationId,
        role: "user",
        content: cleanPrompt,
        thinking: null,
        durationMs: null,
        diffusionStepsJson: null,
      });

      const assistantId = userDbId + 1;

      setMessages((current) => [
        ...current,
        { id: userDbId, role: "user", content: cleanPrompt },
        { id: assistantId, role: "assistant", content: "", pending: true },
      ]);
      setPrompt("");
      setRunning(true);

      // Ensure model is loaded if Agent Mode
      if (chatMode === "agent" && modelStatus !== "ready") {
        setLogs((current) => [
          ...current.slice(-199),
          `[sistema] El modo Agente requiere el modelo en memoria. Iniciando...`,
        ]);
        await invoke("start_model", {
          runnerPath,
          modelPath,
          gpuLayers,
          maxTokens,
          cfgScale: engineKind === "diffusion" ? (cfgScale || null) : null,
          tMin: engineKind === "diffusion" ? (tMin || null) : null,
          tMax: engineKind === "diffusion" ? (tMax || null) : null,
          entropyBound: engineKind === "diffusion" ? (entropyBound || null) : null,
          stability: engineKind === "diffusion" ? (stability || null) : null,
          confidence: engineKind === "diffusion" ? (confidence || null) : null,
        });
        setModelStatus("ready");
      }

      if (modelStatus === "ready" || chatMode === "agent") {
        activeInteractiveMessageId.current = assistantId;
        setLogs((current) => [
          ...current.slice(-199),
          `[sistema] Iniciando generación interactiva instantánea`,
        ]);
        try {
          await invoke("send_interactive_prompt", { prompt: promptToSend });
        } catch (error) {
          const errorContent = `No se pudo enviar el prompt al modelo activo. ${String(error)}`;
          let errorDbId = assistantId;
          try {
            errorDbId = await invoke<number>("db_add_message", {
              conversationId: activeConversationId,
              role: "assistant",
              content: errorContent,
              thinking: null,
              durationMs: null,
              diffusionStepsJson: null,
            });
          } catch (dbErr) {
            console.error("Error al guardar mensaje de error en DB:", dbErr);
          }

          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    id: errorDbId,
                    content: errorContent,
                    pending: false,
                    error: true,
                  }
                : message,
            ),
          );
          activeInteractiveMessageId.current = null;
          setRunning(false);
        }
      } else {
        activeAssistantId.current = assistantId;
        setLogs((current) => [
          ...current.slice(-199),
          `[sistema] Iniciando generación con ${gpuLayers} capas GPU`,
        ]);

        try {
          const result = await invoke<GenerationResult>("generate", {
            request: {
              runner_path: runnerPath,
              model_path: modelPath,
              prompt: cleanPrompt,
              gpu_layers: gpuLayers,
              max_tokens: maxTokens,
              max_steps: engineKind === "diffusion" ? 48 : 0,
              is_diffusion: engineKind === "diffusion",
              cfg_scale: engineKind === "diffusion" ? (cfgScale || null) : null,
              t_min: engineKind === "diffusion" ? (tMin || null) : null,
              t_max: engineKind === "diffusion" ? (tMax || null) : null,
              entropy_bound: engineKind === "diffusion" ? (entropyBound || null) : null,
              stability: engineKind === "diffusion" ? (stability || null) : null,
              confidence: engineKind === "diffusion" ? (confidence || null) : null,
            },
          });
          const answer = result.answer.trim() || "La generaci\u00f3n termin\u00f3 sin texto.";
          const thinking = result.thinking?.trim() || null;

          const msgInMemory = messagesRef.current.find((m) => m.id === assistantId);
          const steps = msgInMemory?.diffusionSteps ?? [];
          const finalSteps = addFinalStep(
            steps,
            result.thinking ? `${result.thinking} <channel|> ${answer}` : answer
          );
          const stepsJson = finalSteps.length > 0 ? JSON.stringify(finalSteps) : null;

          const dbId = await invoke<number>("db_add_message", {
            conversationId: activeConversationId,
            role: "assistant",
            content: answer,
            thinking: thinking,
            durationMs: result.duration_ms ? Math.round(result.duration_ms) : null,
            diffusionStepsJson: stepsJson,
          });

          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    id: dbId,
                    content: answer,
                    thinking: thinking || undefined,
                    durationMs: result.duration_ms,
                    pending: false,
                    diffusionSteps: finalSteps,
                  }
                : message,
            ),
          );
        } catch (error) {
          const errorContent = `No se pudo completar la generación. ${String(error)}`;
          let errorDbId = assistantId;
          try {
            errorDbId = await invoke<number>("db_add_message", {
              conversationId: activeConversationId,
              role: "assistant",
              content: errorContent,
              thinking: null,
              durationMs: null,
              diffusionStepsJson: null,
            });
          } catch (dbErr) {
            console.error("Error al guardar mensaje de error en DB:", dbErr);
          }

          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    id: errorDbId,
                    content: errorContent,
                    pending: false,
                    error: true,
                  }
                : message,
            ),
          );
        } finally {
          activeAssistantId.current = null;
          setRunning(false);
          refreshStatus();
          window.setTimeout(() => textareaRef.current?.focus(), 0);
        }
      }
    } catch (e) {
      console.error("Error al enviar mensaje:", e);
    }
  }

  async function stopGeneration() {
    try {
      if (modelStatus === "ready") {
        await invoke("stop_model");
        setModelStatus("stopped");
      } else {
        await invoke("stop_generation");
      }
      setLogs((current) => [
        ...current.slice(-199),
        "[sistema] Generación detenida",
      ]);
    } finally {
      activeInteractiveMessageId.current = null;
      setRunning(false);
    }
  }

  async function startNewConversation() {
    if (running) return;
    await createNewConversation("Nueva conversación");
  }

  function handleComposerKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendPrompt();
    }
  }

  async function generateImage() {
    const cleanPrompt = imagePrompt.trim();
    if (!cleanPrompt || imageGenerating) return;

    setImageGenerating(true);
    setImageProgress(null);

    const seed = imageSeed > 0 ? imageSeed : Math.floor(Math.random() * 2147483647);

    setLogs((current) => [
      ...current.slice(-199),
      `[sistema] Generando imagen ${imageWidth}x${imageHeight} · ${imageSteps} pasos`,
    ]);

    try {
      const result = await invoke<ImageResult>("generate_image", {
        request: {
          runner_path: imageRunnerPath,
          model_path: imageModelPath,
          prompt: cleanPrompt,
          negative_prompt: negativePrompt,
          width: imageWidth,
          height: imageHeight,
          steps: imageSteps,
          guidance: imageGuidance,
          seed,
          extra_args: "",
          lora_dir: imageLoraDir,
        },
      });

      let dbId = imageNextId.current++;
      if (activeConversationId !== null) {
        try {
          dbId = await invoke<number>("db_add_image", {
            conversationId: activeConversationId,
            imageBase64: result.image_base64,
            prompt: cleanPrompt,
            negativePrompt,
            width: imageWidth,
            height: imageHeight,
            steps: imageSteps,
            guidance: imageGuidance,
            seed,
            durationMs: Math.round(result.duration_ms),
            timestamp: Date.now(),
          });
        } catch (dbErr) {
          console.error("Error al guardar imagen en DB:", dbErr);
        }
      }

      const newImage: GeneratedImage = {
        id: dbId,
        base64: result.image_base64,
        prompt: cleanPrompt,
        negativePrompt,
        width: imageWidth,
        height: imageHeight,
        steps: imageSteps,
        guidance: imageGuidance,
        seed,
        durationMs: result.duration_ms,
        timestamp: Date.now(),
      };
      setGeneratedImages((current) => [newImage, ...current]);
      setSelectedImageId(newImage.id);
      setLogs((current) => [
        ...current.slice(-199),
        `[sistema] Imagen generada en ${(result.duration_ms / 1000).toFixed(1)}s`,
      ]);
    } catch (error) {
      setLogs((current) => [
        ...current.slice(-199),
        `[error] No se pudo generar la imagen: ${String(error)}`,
      ]);
    } finally {
      setImageGenerating(false);
      setImageProgress(null);
    }
  }

  async function stopImageGeneration() {
    try {
      await invoke("stop_image_generation");
      setLogs((current) => [
        ...current.slice(-199),
        "[sistema] Generación de imagen detenida",
      ]);
    } finally {
      setImageGenerating(false);
      setImageProgress(null);
    }
  }

  const vramPercent = status.vram_total_mb
    ? Math.min(100, (status.vram_used_mb / status.vram_total_mb) * 100)
    : 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-logo-container" aria-hidden="true">
            <LogoIcon />
          </div>
          <div>
            <strong>Model Studio</strong>
            <span>Local inference</span>
          </div>
        </div>
        <div className="engine-selector" aria-label="Tipo de motor">
          <span className="engine-label">Motor</span>
          <div className="engine-options">
            <button
              className={engineKind === "diffusion" ? "active" : ""}
              type="button"
              aria-pressed={engineKind === "diffusion"}
              onClick={() => setEngineKind("diffusion")}
            >
              <DiffusionIcon />
              <span>
                <strong>Texto</strong>
                <small>Difusión discreta</small>
              </span>
            </button>
            <button
              className={engineKind === "llm" ? "active" : ""}
              type="button"
              aria-pressed={engineKind === "llm"}
              onClick={() => setEngineKind("llm")}
            >
              <LlmIcon />
              <span>
                <strong>LLM</strong>
                <small>Autoregresivo</small>
              </span>
            </button>
            <button
              className={engineKind === "image" ? "active" : ""}
              type="button"
              aria-pressed={engineKind === "image"}
              onClick={() => setEngineKind("image")}
            >
              <ImageIcon />
              <span>
                <strong>Imagen</strong>
                <small>Difusión visual</small>
              </span>
            </button>
          </div>
        </div>
        <div className="topbar-status">
          <span className="gpu-tag" title={`GPU ${status.gpu_vendor || "—"}`}>
            <GpuIcon />
            {status.gpu_detected ? status.gpu_name : "Sin GPU"}
          </span>
          <span className={`status ${statusKind}`}>
            <i aria-hidden="true" />
            {readiness}
          </span>
          {running ? (
            <button className="button danger" onClick={stopGeneration}>
              <StopIcon />
              Detener
            </button>
          ) : (
            <button className="button secondary" onClick={refreshStatus}>
              <RefreshIcon />
              Verificar
            </button>
          )}
        </div>
      </header>

      <aside className="sidebar">
        {/* Project Workspace Selector */}
        <div className="project-workspace-selector">
          <span className="project-selector-label">Proyecto / Espacio</span>
          <div className="project-select-row">
            <select
              className="project-select"
              value={activeProjectId ?? ""}
              onChange={(e) => {
                const val = e.target.value;
                setActiveProjectId(val ? Number(val) : null);
              }}
              disabled={running}
            >
              <option value="">🌐 Global (Sin Proyecto)</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  📁 {p.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="add-project-btn"
              onClick={() => setShowProjectModal(true)}
              disabled={running}
              title="Añadir proyecto local (carpeta)"
            >
              +
            </button>
            {activeProjectId !== null && (
              <button
                type="button"
                className="delete-project-btn"
                onClick={() => deleteProject(activeProjectId)}
                disabled={running}
                title="Eliminar proyecto"
              >
                🗑️
              </button>
            )}
          </div>
        </div>

        <button
          className="new-chat"
          onClick={startNewConversation}
          disabled={running}
          aria-label="Crear conversación nueva"
        >
          <span aria-hidden="true">+</span>
          Nueva conversación
        </button>

        {/* Mode Selector and Agent Selectors */}
        <div className="mode-selector-container">
          <span className="mode-selector-label">Modo Operativo</span>
          <div className="mode-selector-buttons">
            <button
              type="button"
              className={`mode-btn ${chatMode === 'chat' ? 'active' : ''}`}
              onClick={() => setChatMode('chat')}
              disabled={running}
            >
              Chat
            </button>
            <button
              type="button"
              className={`mode-btn ${chatMode === 'agent' ? 'active' : ''}`}
              onClick={() => setChatMode('agent')}
              disabled={running}
            >
              Agente
            </button>
            <button
              type="button"
              className={`mode-btn ${chatMode === 'super-agent' ? 'active' : ''}`}
              onClick={() => setChatMode('super-agent')}
              disabled={running}
            >
              Súper
            </button>
          </div>

          {chatMode === 'agent' && (
            <div className="agent-subselector">
              <select
                className="agent-select-input"
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value as any)}
                disabled={running}
              >
                <option value="developer">👨‍💻 Programador (Developer)</option>
                <option value="researcher">🔍 Investigador (Researcher)</option>
                <option value="file-specialist">📂 Archivos (File Specialist)</option>
              </select>
            </div>
          )}
        </div>
        <nav aria-label="Conversaciones">
          <p className="nav-label">
            <span>Recientes</span>
          </p>
          {conversations.map((conv) => {
            const isActive = conv.id === activeConversationId;
            const isEditing = conv.id === editingConversationId;
            
            return (
              <div
                key={conv.id}
                className={`conversation-item-container${isActive ? " active" : ""}`}
              >
                {isEditing ? (
                  <input
                    type="text"
                    className="conv-rename-input"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => handleRenameKeyDown(e, conv.id)}
                    onBlur={() => saveConversationTitle(conv.id)}
                    autoFocus
                  />
                ) : (
                  <button
                    type="button"
                    className={`conversation-link${isActive ? " active" : ""}`}
                    onClick={() => selectConversation(conv.id)}
                  >
                    <span className="conv-icon" aria-hidden="true">
                      <ChatIcon />
                    </span>
                    <span className="conv-text">
                      {conv.title}
                    </span>
                  </button>
                )}
                
                {!isEditing && (
                  <div className="conv-actions">
                    <button
                      type="button"
                      onClick={(e) => startEditingConversation(conv.id, conv.title, e)}
                      title="Renombrar conversación"
                      className="conv-action-btn"
                    >
                      <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor">
                        <path d="M12.854.146a.5.5 0 0 0-.707 0L10.5 1.793 14.207 5.5l1.647-1.646a.5.5 0 0 0 0-.708l-3-3zm.646 6.061L9.793 2.5 3.293 9H3.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.207l6.5-6.5zm-7.468 7.468A.5.5 0 0 1 6 13.5V13h-.5a.5.5 0 0 1-.5-.5V12h-.5a.5.5 0 0 1-.5-.5V11h-.5a.5.5 0 0 1-.5-.5V10h-.5a.499.499 0 0 1-.175-.03l-.179.178c-.098.098-.241.14-.379.11L1 10.707V12h1v.5a.5.5 0 0 1 .5.5v.5h1v.5a.5.5 0 0 1 .5.5v.5h1v.5a.5.5 0 0 1 .5.5v.207l.3-.3z"/>
                      </svg>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => deleteConversation(conv.id, e)}
                      title="Eliminar conversación"
                      className="conv-action-btn hover-danger"
                    >
                      <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor">
                        <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
                        <path fillRule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
                      </svg>
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <div className="local-note">
            <span className="dot" aria-hidden="true" />
            Todo se procesa localmente
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--muted)", fontSize: "10.5px", fontWeight: 600, letterSpacing: "0.02em", marginTop: "8px" }}>
            <span style={{ position: "relative", width: "8px", height: "8px", borderRadius: "50%", background: "#4f46e5", boxShadow: "0 0 8px #4f46e5", flexShrink: 0 }} />
            <span>API Server: <code style={{ fontFamily: "Consolas, monospace", background: "rgba(255,255,255,0.06)", padding: "1px 4px", borderRadius: "3px", fontSize: "10px", color: "#818cf8" }}>http://127.0.0.1:1234</code></span>
          </div>
        </div>
      </aside>

      {engineKind !== "image" ? (
      <section className="chat">
        <div className="conversation" ref={conversationRef}>
          <div className="conversation-inner">
            <div className="conversation-hero">
              <div>
                <h1>
                  <span className="hero-icon" aria-hidden="true">
                    <DiffusionIcon />
                  </span>
                  {conversations.find((c) => c.id === activeConversationId)?.title || "Prueba local"}
                </h1>
                <p>
                  DiffusionGemma 26B A4B
                  <span className="meta-pill">Q4_K_M</span>
                  <span className="meta-pill">26B parámetros</span>
                </p>
              </div>
              <span className="architecture-label">
                <DiffusionIcon />
                Modelo de difusión discreta
              </span>
            </div>

            {messages.length <= 1 && !running && (
              <WelcomeState
                onSuggestion={(text) => {
                  setPrompt(text);
                  window.setTimeout(() => textareaRef.current?.focus(), 0);
                }}
              />
            )}

            <div className="message-list">
              {messages.map((message) =>
                message.role === "user" ? (
                  <article className="message user" key={message.id}>
                    <div className="user-bubble">{message.content}</div>
                  </article>
                ) : (
                  <article className="message assistant" key={message.id}>
                    <div className={`assistant-mark${message.pending ? " pending" : ""}`} aria-hidden="true">
                      <ModelGlyph />
                    </div>
                    <div className="assistant-body">
                      {!message.pending && (
                        <span className="assistant-name">
                          DiffusionGemma
                          <span className="model-tag">26B A4B</span>
                        </span>
                      )}
                      {message.pending ? (
                        <>
                          <div className="thinking-live" role="status">
                            <span className="thinking-bars" aria-hidden="true">
                              <span />
                              <span />
                              <span />
                            </span>
                            <span>
                              Pensando
                              {elapsedSeconds > 0 ? ` durante ${elapsedSeconds} s` : ""}
                            </span>
                          </div>
                          {message.diffusionSteps && message.diffusionSteps.length > 0 && (
                            <DiffusionCanvas steps={message.diffusionSteps} live />
                          )}
                          {message.superAgentSteps && message.superAgentSteps.length > 0 && (
                             <SuperAgentPanel
                               steps={message.superAgentSteps}
                               messageId={message.id}
                               expandedPanels={expandedPanels}
                               onToggle={(id) =>
                                 setExpandedPanels((prev) => ({ ...prev, [id]: !prev[id] }))
                               }
                             />
                           )}
                          {message.content && (
                            <div className="streaming-content" style={{ marginTop: "14px" }}>
                              <MarkdownMessage content={message.content} />
                            </div>
                          )}
                        </>
                      ) : (
                        <>
                          {message.diffusionSteps && message.diffusionSteps.length > 0 && (
                            <details className="diffusion-details">
                              <summary>
                                <span className="summary-dot" aria-hidden="true" />
                                <span>
                                  Proceso de difusión · {message.diffusionSteps.length} pasos
                                </span>
                                <ChevronIcon />
                              </summary>
                              <DiffusionCanvas steps={message.diffusionSteps} />
                            </details>
                          )}
                          {message.superAgentSteps && message.superAgentSteps.length > 0 && (
                             <SuperAgentPanel
                               steps={message.superAgentSteps}
                               messageId={message.id}
                               expandedPanels={expandedPanels}
                               onToggle={(id) =>
                                 setExpandedPanels((prev) => ({ ...prev, [id]: !prev[id] }))
                               }
                             />
                           )}
                          {message.thinking && (
                            <ThinkingBlock
                              thinking={message.thinking}
                              durationMs={message.durationMs}
                            />
                          )}
                          <MarkdownMessage
                            content={message.content}
                            error={message.error}
                          />
                        </>
                      )}
                    </div>
                  </article>
                ),
              )}
            </div>
          </div>
        </div>

        <form className="composer" onSubmit={sendPrompt}>
          <textarea
            ref={textareaRef}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={handleComposerKey}
            placeholder={running ? "DiffusionGemma está pensando…" : "Escribe un mensaje para DiffusionGemma…"}
            rows={3}
            disabled={running}
            aria-label="Mensaje para DiffusionGemma"
          />
          <div className="composer-footer">
            <div className="composer-tools">
              <button type="button" className="composer-tool" aria-label="Adjuntar archivo" title="Adjuntar">
                <AttachIcon />
              </button>
              <button type="button" className="composer-tool active" aria-label="Modo difusión" title="Modo difusión">
                <SparkleIcon />
              </button>
            </div>
            <span className="hint">
              <kbd>Enter</kbd> enviar · <kbd>Shift</kbd>+<kbd>Enter</kbd> nueva línea
            </span>
            {running ? (
              <button type="button" className="button danger" onClick={stopGeneration}>
                <StopIcon />
                Detener
              </button>
            ) : (
              <button
                type="submit"
                className="button primary"
                disabled={!prompt.trim() || !status.gpu_detected || !status.runner_exists || !status.model_exists}
              >
                <SendIcon />
                Enviar
              </button>
            )}
          </div>
        </form>
      </section>
      ) : (
        <ImageStudio
          imagePrompt={imagePrompt}
          setImagePrompt={setImagePrompt}
          negativePrompt={negativePrompt}
          setNegativePrompt={setNegativePrompt}
          imageWidth={imageWidth}
          setImageWidth={setImageWidth}
          imageHeight={imageHeight}
          setImageHeight={setImageHeight}
          imageSteps={imageSteps}
          setImageSteps={setImageSteps}
          imageGuidance={imageGuidance}
          setImageGuidance={setImageGuidance}
          imageSeed={imageSeed}
          setImageSeed={setImageSeed}
          imageGenerating={imageGenerating}
          imageProgress={imageProgress}
          generatedImages={generatedImages}
          selectedImageId={selectedImageId}
          setSelectedImageId={setSelectedImageId}
          onGenerate={generateImage}
          onStop={stopImageGeneration}
        />
      )}

      <aside className="diagnostics">
        <section>
          <div className="panel-title">
            <h2>{engineKind === "image" ? "Imagen" : "Ejecución"}</h2>
            <div className="panel-actions">
              {engineKind !== "image" && (
                <button
                type="button"
                className={`button-model-status ${modelStatus}`}
                onClick={toggleModelPersistent}
                disabled={modelStatus === "loading" || running || (modelStatus === "stopped" && !status.gpu_detected)}
                title={
                  modelStatus === "ready" 
                    ? "Detener modelo (Liberar VRAM)" 
                    : status.gpu_detected
                      ? "Iniciar modelo (Mantener cargado en VRAM)"
                      : "GPU no detectada"
                }
              >
                {modelStatus === "loading" && (
                  <>
                    <span className="ms-dot" />
                    Cargando
                  </>
                )}
                {modelStatus === "ready" && (
                  <>
                    <span className="ms-dot" />
                    Activo
                  </>
                )}
                {modelStatus === "stopped" && (
                  <>
                    <PowerIcon />
                    Iniciar
                  </>
                )}
              </button>
              )}
              <button
                type="button"
                className="icon-button"
                onClick={refreshStatus}
                aria-label="Actualizar diagnóstico"
                title="Actualizar"
              >
                <RefreshIcon />
              </button>
            </div>
          </div>
          {engineKind === "image" ? (
            <>
              <label>
                Runner de imagen
                <input value={imageRunnerPath} onChange={(event) => setImageRunnerPath(event.target.value)} />
              </label>
              <label>
                Modelo de imagen
                <input value={imageModelPath} onChange={(event) => setImageModelPath(event.target.value)} placeholder="Ruta al modelo .gguf o .safetensors" />
              </label>
              <label>
                Directorio de LoRAs
                <input value={imageLoraDir} onChange={(event) => setImageLoraDir(event.target.value)} placeholder="Ej: C:\ComfyUI\models\loras" />
              </label>
              <div className="image-runner-hint">
                <p>
                  Configura un runner de generación de imágenes (ej.
                  stable-diffusion.cpp) y la ruta a un modelo de difusión
                  visual. El runner debe aceptar:{" "}
                  <code>-m modelo -p prompt -W ancho -H alto -s pasos -o salida.png</code>
                </p>
              </div>
            </>
          ) : (
            <>
          <label>
            Runner
            <input value={runnerPath} onChange={(event) => setRunnerPath(event.target.value)} />
          </label>
          <label>
            Modelo
            <input value={modelPath} onChange={(event) => setModelPath(event.target.value)} />
          </label>
          <div className="control-row">
            <label>
              Capas GPU
              <input
                type="number"
                min="0"
                max="99"
                value={gpuLayers}
                onChange={(event) => setGpuLayers(Number(event.target.value))}
              />
            </label>
            <label>
              Tokens
              <input
                type="number"
                min="64"
                max="4096"
                step="64"
                value={maxTokens}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </label>
          </div>
          
          <details className="advanced-settings-details" style={{ marginTop: "14px" }}>
            <summary className="advanced-settings-summary" style={{ cursor: "pointer", fontSize: "11px", color: "var(--muted)", fontWeight: "bold", textTransform: "uppercase", letterSpacing: "0.5px" }}>Ajustes avanzados</summary>
            <div className="advanced-settings-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginTop: "10px" }}>
              <label style={{ fontSize: "11px", color: "var(--muted)" }}>
                CFG Scale
                <input
                  type="number"
                  min="0"
                  max="10"
                  step="0.1"
                  value={cfgScale}
                  onChange={(event) => setCfgScale(Number(event.target.value))}
                  style={{ width: "100%", padding: "5px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--fg)" }}
                />
              </label>
              <label style={{ fontSize: "11px", color: "var(--muted)" }}>
                Estabilidad
                <input
                  type="number"
                  min="1"
                  max="10"
                  step="1"
                  value={stability}
                  onChange={(event) => setStability(Number(event.target.value))}
                  style={{ width: "100%", padding: "5px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--fg)" }}
                />
              </label>
              <label style={{ fontSize: "11px", color: "var(--muted)" }}>
                t_min (Temp mín)
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.05"
                  value={tMin}
                  onChange={(event) => setTMin(Number(event.target.value))}
                  style={{ width: "100%", padding: "5px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--fg)" }}
                />
              </label>
              <label style={{ fontSize: "11px", color: "var(--muted)" }}>
                t_max (Temp máx)
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.05"
                  value={tMax}
                  onChange={(event) => setTMax(Number(event.target.value))}
                  style={{ width: "100%", padding: "5px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--fg)" }}
                />
              </label>
              <label style={{ fontSize: "11px", color: "var(--muted)" }}>
                Límite Entropía
                <input
                  type="number"
                  min="0"
                  max="5"
                  step="0.05"
                  value={entropyBound}
                  onChange={(event) => setEntropyBound(Number(event.target.value))}
                  style={{ width: "100%", padding: "5px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--fg)" }}
                />
              </label>
              <label style={{ fontSize: "11px", color: "var(--muted)" }}>
                Parada (conf)
                <input
                  type="number"
                  min="0.0001"
                  max="0.1"
                  step="0.001"
                  value={confidence}
                  onChange={(event) => setConfidence(Number(event.target.value))}
                  style={{ width: "100%", padding: "5px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--fg)" }}
                />
              </label>
            </div>
          </details>

          {/* Custom Tools Section */}
          <div className="custom-tools-config-section" style={{ marginTop: "20px", paddingTop: "14px", borderTop: "1px solid var(--border-soft)" }}>
            <span className="mode-selector-label" style={{ display: "block", marginBottom: "8px" }}>Herramientas de Agente</span>
            <button
              type="button"
              className="manage-tools-btn"
              onClick={() => setShowToolModal(true)}
              style={{
                width: "100%",
                padding: "8px",
                borderRadius: "8px",
                border: "1px solid var(--border-soft)",
                background: "var(--inset)",
                color: "var(--text-main)",
                fontSize: "11px",
                fontWeight: "600",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "6px"
              }}
            >
              🛠️ Crear / Ver Herramientas
            </button>
          </div>
            </>
          )}
        </section>

        <section>
          <div className="panel-title">
            <h2>GPU</h2>
            <div className="panel-actions">
              <button
                type="button"
                className="icon-button"
                onClick={refreshStatus}
                aria-label="Actualizar GPU"
                title="Actualizar"
              >
                <RefreshIcon />
              </button>
            </div>
          </div>
          <div className={`gpu-card${!status.gpu_detected ? " no-gpu" : ""}${status.gpu_detected && !status.gpu_supports_metrics ? " no-metrics" : ""}`}>
            <div className="gpu-headline">
              <div className="gpu-name">
                <span className={`gpu-icon${!status.gpu_detected ? " warn" : ""}`} aria-hidden="true">
                  {status.gpu_detected ? <GpuIcon /> : <WarningIcon />}
                </span>
                <div className="gpu-text">
                  <strong>{status.gpu_name}</strong>
                  <small>
                    {status.gpu_detected
                      ? status.gpu_vendor === "Desconocido"
                        ? "GPU detectada"
                        : status.gpu_vendor
                      : "No detectada"}
                  </small>
                </div>
              </div>
              {status.gpu_detected && status.gpu_vendor && status.gpu_vendor !== "—" && (
                <span className="vendor-badge">{status.gpu_vendor}</span>
              )}
            </div>

            {!status.gpu_detected ? (
              <div className="gpu-warning">
                <p>
                  No se detectó una GPU dedicada. El runner requiere una GPU
                  compatible (NVIDIA CUDA) para inferencia local.
                </p>
              </div>
            ) : !status.gpu_supports_metrics ? (
              <div className="gpu-warning">
                <p>
                  GPU detectada pero sin telemetría en vivo. Las métricas de VRAM,
                  uso y temperatura no están disponibles para este controlador.
                </p>
              </div>
            ) : (
              <>
                <div className="metric-line">
                  <span className="label">Memoria VRAM</span>
                  <span className="value">
                    {(status.vram_used_mb / 1024).toFixed(1)} /{" "}
                    {(status.vram_total_mb / 1024).toFixed(1)} GB
                  </span>
                </div>
                <div
                  className={`meter ${
                    vramPercent > 85 ? "crit" : vramPercent > 65 ? "warn" : ""
                  }`}
                >
                  <span style={{ width: `${vramPercent}%` }} />
                </div>
                <div className="metric-grid">
                  <div className="metric-tile">
                    <span className="tile-label">
                      <UsageIcon />
                      Uso
                    </span>
                    <span className="tile-value">{status.gpu_utilization}%</span>
                    <span className="tile-bar" style={{ width: `${status.gpu_utilization}%` }} />
                  </div>
                  <div
                    className={`metric-tile ${
                      status.temperature >= 80
                        ? "crit"
                        : status.temperature >= 70
                          ? "warn"
                          : ""
                    }`}
                  >
                    <span className="tile-label">
                      <TempIcon />
                      Temp
                    </span>
                    <span className="tile-value">{status.temperature}°C</span>
                    <span
                      className="tile-bar"
                      style={{ width: `${Math.min(100, (status.temperature / 95) * 100)}%` }}
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        </section>

        <section className="log-section">
          <div className="panel-title">
            <h2>Registro</h2>
            <button onClick={() => setLogs([])}>Limpiar</button>
          </div>
          <div className="log-console" aria-live="polite">
            {logs.length > 0 ? (
              logs.map((log, index) => {
                let logClass = "log-line";
                if (log.startsWith("[sistema]") || log.startsWith("[system]")) logClass += " system";
                else if (log.startsWith("[stdout]")) logClass += " stdout";
                else if (log.startsWith("[stderr]")) logClass += " stderr";
                else if (log.startsWith("[error]")) logClass += " error";
                return (
                  <div className={logClass} key={index}>
                    {log}
                  </div>
                );
              })
            ) : (
              <div className="log-line empty">Sin eventos.</div>
            )}
          </div>
        </section>
      </aside>

      {/* Modal: Crear Proyecto */}
      {showProjectModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3>Crear Nuevo Proyecto</h3>
              <button className="close-modal-btn" onClick={() => setShowProjectModal(false)}>×</button>
            </div>
            <form onSubmit={handleCreateProject} className="modal-form">
              <label>
                Nombre del Proyecto
                <input
                  type="text"
                  placeholder="ej. Mi Web App"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  required
                />
              </label>
              <label>
                Ruta Local de la Carpeta
                <input
                  type="text"
                  placeholder="ej. C:\proyectos\mi-app"
                  value={newProjectPath}
                  onChange={(e) => setNewProjectPath(e.target.value)}
                  required
                />
              </label>
              <div className="modal-actions">
                <button type="button" onClick={() => setShowProjectModal(false)}>Cancelar</button>
                <button type="submit" className="btn-primary">Crear Proyecto</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Gestor de Herramientas Personalizadas */}
      {showToolModal && (
        <div className="modal-overlay">
          <div className="modal-content tool-manager-modal">
            <div className="modal-header">
              <h3>Herramientas de Agente</h3>
              <button className="close-modal-btn" onClick={() => setShowToolModal(false)}>×</button>
            </div>
            
            <div className="modal-section">
              <h4>Crear Nueva Herramienta</h4>
              <form onSubmit={handleCreateTool} className="modal-form">
                <label>
                  Nombre de la Herramienta (solo letras, números, _ o -)
                  <input
                    type="text"
                    placeholder="ej. obtener_clima"
                    value={newToolName}
                    onChange={(e) => setNewToolName(e.target.value)}
                    required
                  />
                </label>
                <label>
                  Descripción
                  <input
                    type="text"
                    placeholder="ej. Devuelve el clima actual para una ciudad."
                    value={newToolDescription}
                    onChange={(e) => setNewToolDescription(e.target.value)}
                    required
                  />
                </label>
                <label>
                  Plantilla de Comando Shell (usa {"{{arg}}"} para inyectar argumentos)
                  <input
                    type="text"
                    placeholder="ej. curl -s https://wttr.in/{{arg}}"
                    value={newToolCommandTemplate}
                    onChange={(e) => setNewToolCommandTemplate(e.target.value)}
                    required
                  />
                </label>
                <button type="submit" className="btn-primary" style={{ marginTop: "10px" }}>
                  Agregar Herramienta
                </button>
              </form>
            </div>

            <div className="modal-section" style={{ marginTop: "20px", borderTop: "1px solid var(--border-soft)", paddingTop: "14px" }}>
              <h4>Herramientas Existentes</h4>
              <div className="tool-list">
                {customTools.length > 0 ? (
                  customTools.map((tool) => (
                    <div key={tool.id} className="tool-item" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px", background: "var(--inset)", border: "1px solid var(--border-soft)", borderRadius: "8px", marginBottom: "8px" }}>
                      <div>
                        <strong>TOOL: {tool.name}</strong>
                        <p style={{ margin: "2px 0 0", fontSize: "10.5px", color: "var(--muted)" }}>{tool.description}</p>
                        <code style={{ fontSize: "10px", color: "var(--accent)" }}>{tool.command_template}</code>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDeleteTool(tool.id)}
                        style={{ background: "transparent", border: "none", color: "var(--error)", cursor: "pointer", fontSize: "14px" }}
                        title="Eliminar herramienta"
                      >
                        🗑️
                      </button>
                    </div>
                  ))
                ) : (
                  <p style={{ fontSize: "11px", color: "var(--muted)" }}>No hay herramientas personalizadas creadas.</p>
                )}
              </div>
            </div>
            
            <div className="modal-actions" style={{ marginTop: "14px" }}>
              <button type="button" onClick={() => setShowToolModal(false)}>Cerrar</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;
