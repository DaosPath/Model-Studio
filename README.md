# <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><path d="M4.5 16.5c-1.5 1.26-2.5 3.19-2.5 5.5h20c0-2.31-1-4.24-2.5-5.5"/><path d="M12 2C7.57 2 4 5.57 4 10c0 4.7 3.3 8.3 8 8s8-3.3 8-8c0-4.43-3.57-8-8-8z"/><path d="M12 6v6"/><path d="M9 9h6"/></svg>Model Studio

High-performance desktop application designed with **Tauri, React, TypeScript, and Rust** for local execution, control, and automation of language models (LLMs in GGUF format) and image diffusion models.

Model Studio allows you to run advanced artificial intelligence 100% privately and locally, isolated from the cloud, with a premium interface and professional workflows.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#eab308" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>Key Features

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ec4899" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2Z"/></svg>Local LLM & Diffusion Engine
* **High-Performance Inference**: GPU-accelerated execution of GGUF models with dynamic layer offloading.
* **Integrated Stable Diffusion**: Direct image generation in the interface using local diffusion runners.
* **Live Hyperparameters**: Full control over inference parameters (Temperature, Top-P, Repeat Penalty, and Stop Sequences) directly from the **Advanced Settings** panel.
* **Token Saver (KV Cache Optimizer)**: Real-time token tracking with automatic history compression and summarization in the background when a conversation exceeds 3,000 tokens.

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><rect width="18" height="12" x="3" y="8" rx="2" ry="2"/><path d="M12 2v6"/><path d="M8 2h8"/><path d="M18 16a3 3 0 0 0-6 0h6Z"/><path d="M6 13h.01"/><path d="M18 13h.01"/></svg>Intelligent Agents (ReAct & Multi-Agent)
* **Agent Mode (ReAct)**: Autonomous agents that inspect the local directory context and execute `TOOL:` calls in a loop until the user's task is resolved.
* **Super Agent Mode (Multi-Agent)**: A sequential pipeline of 5 specialized subagents (*Analyzer*, *Critic*, *Researcher*, *Validator*, and *Synthesizer*) featuring an animated progress dashboard.
* **Customizable System Prompts**: Dedicated configuration panel to edit, save in SQLite, and restore system prompts for each agent.

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>Local Agent Tools
* **File Management**: Direct read and recursive directory listing tools with workspace directory isolation.
* **Stateful Navigation**: Persistent `cd` tool per conversation allowing the agent to logically change the active working directory.
* **Streaming Terminal**: Built-in console in the UI executing PowerShell (Windows) or sh (Unix) commands asynchronously with termination (`kill`) control.
* **Python Sandbox**: Local script execution. Matplotlib `.png` charts and plots are automatically detected and rendered in the chat flow.
* **Smart Editing**: File editing via direct overwrite or incremental patches (`patch_file`) utilizing `SEARCH/REPLACE` blocks for fast and safe modifications.
* **Custom Tools Builder**: Interface to register custom system commands as agent tools using templates (e.g., `curl https://wttr.in/{{arg}}`).

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>Local Code Search (RAG)
* **TF-IDF Algorithm**: Index-based local code search written in Rust that retrieves relevant code snippets in milliseconds.
* **Advanced Filtering**: Support for segmenting searches using file extensions (`ext:ts`) and subdirectories (`path:utils`).

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z"/></svg>Live Code Editor & Code Application
* **Integrated Editor**: Live file viewer and code editor in a modal with synchronized line numbers and direct saving to disk.
* **PrismJS Highlighting**: Rich syntax highlighting with premium tomorrow-dark styling for JS, TS, Rust, Python, C++, Bash, and more.
* **Apply to Editor**: Dynamic button on code snippets to instantly apply and overwrite the active live code editor content.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>Architecture & Tech Stack

* **Frontend**: React 18, TypeScript, Vite, Vanilla CSS (designed with responsive grids, glassmorphism, and micro-animations).
* **Syntax Color**: PrismJS with *Tomorrow-Tomorrow* dark theme.
* **Backend**: Rust, Tauri v1 (handling subprocesses, SQLite queries, and native command integrations).
* **Database**: SQLite (via `rusqlite` in Rust) to persist projects, conversations, messages, custom prompts, and custom tools.
* **Local API**: OpenAI-compatible endpoint running on port `1234`.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect width="20" height="12" x="2" y="3" rx="2"/><line x1="12" x2="12" y1="15" y2="21"/><line x1="8" x2="16" y1="21" y2="21"/></svg>System Requirements

* **Operating System**: Windows (PowerShell enabled) or Unix (Linux/macOS).
* **Node.js**: v18 or higher.
* **Package Manager**: `pnpm` installed globally.
* **Rust**: Rust toolchain installed (`rustc`, `cargo`, and `rustup`).
* **Inference Executables**: Local inference binaries (`llama-diffusion-cli.exe` or equivalent LLM/diffusion runners) placed inside the `runtime/` folder.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 2v16M19 11l-7 7-7-7M2 22h20"/></svg>Installation & Setup

Follow these steps to compile and run Model Studio locally in development mode:

1. **Clone the repository** and open the project directory.
2. **Install frontend dependencies**:
   ```powershell
   pnpm install
   ```
3. **Prepare Runtime Directory**:
   Place your inference CLI binaries and their required DLLs inside the `runtime/` directory at the project root.
4. **Set Up Local Models**:
   Create a `models/` directory and place your `.gguf` files or diffusion checkpoints inside.
5. **Start Development App**:
   ```powershell
   pnpm tauri dev
   ```

Tauri will compile the Rust backend, launch the Vite local server, and open the Model Studio GUI window on your desktop.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.1a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>Advanced Configuration

### SQLite Database
The application automatically creates and updates a local database file to manage:
* The `projects` and `conversations` tables (for workspace chat history isolation).
* The `agent_prompts` table to persist customized agent system instructions.
* The `custom_tools` database entries created via the sidebar GUI.

### Local API Server
Upon launch, Model Studio runs an OpenAI-compliant server at `http://127.0.0.1:1234`. You can connect editor extensions or external UI wrappers to this local API using any standard client SDK.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>License & Authorship

Developed under premium standards of user experience and software architecture by **DaosPath**. All rights reserved.
