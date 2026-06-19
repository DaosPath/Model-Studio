# <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><path d="M4.5 16.5c-1.5 1.26-2.5 3.19-2.5 5.5h20c0-2.31-1-4.24-2.5-5.5"/><path d="M12 2C7.57 2 4 5.57 4 10c0 4.7 3.3 8.3 8 8s8-3.3 8-8c0-4.43-3.57-8-8-8z"/><path d="M12 6v6"/><path d="M9 9h6"/></svg>Model Studio

Aplicación de escritorio premium de alto rendimiento diseñada con **Tauri, React, TypeScript y Rust** para la ejecución local, control y automatización de modelos de lenguaje (LLMs en formato GGUF) y modelos de difusión de imágenes.

Model Studio te permite ejecutar inteligencia artificial avanzada de forma 100% privada y local, aislada de la nube, con una interfaz premium y flujos de trabajo profesionales.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#eab308" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>Características Principales

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ec4899" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2Z"/></svg>Motor de LLM y Difusión Local
* **Inferencia de Alto Rendimiento**: Ejecución de modelos en formato GGUF delegando capas dinámicamente a la GPU.
* **Stable Diffusion integrado**: Generación de imágenes directa en la interfaz mediante runners de difusión local.
* **Hiperparámetros en Vivo**: Control absoluto de la inferencia (Temperatura, Top-P, Penalización de repetición y Secuencias de parada) directo desde el panel de **Ajustes avanzados**.
* **Token Saver (KV Cache Optimizer)**: Monitoreo de tokens en tiempo real con compresión automática y resumido en segundo plano del historial cuando la conversación supera los 3,000 tokens para ahorrar memoria.

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><rect width="18" height="12" x="3" y="8" rx="2" ry="2"/><path d="M12 2v6"/><path d="M8 2h8"/><path d="M18 16a3 3 0 0 0-6 0h6Z"/><path d="M6 13h.01"/><path d="M18 13h.01"/></svg>Agentes Inteligentes (ReAct & Multiagente)
* **Modo Agente (ReAct)**: Agentes autónomos que evalúan el contexto local y ejecutan llamadas del tipo `TOOL:` en bucle hasta resolver el problema del usuario.
* **Modo Súper Agente (Multiagente)**: Pipeline secuencial compuesto por 5 subagentes especializados (*Analista*, *Crítico*, *Investigador*, *Validador* y *Sintetizador*) con interfaz de progreso animada.
* **System Prompts Personalizables**: Pestaña dedicada para editar, guardar en base de datos y restaurar las directrices de sistema de cada agente.

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>Caja de Herramientas Local (Agent Tools)
* **Gestión de Archivos**: Lectura y listado recursivo de directorios con aislamiento a nivel de proyecto.
* **Navegación con Estado**: Herramienta `cd` persistente por chat que permite al agente cambiar el directorio de trabajo actual de forma lógica.
* **Terminal en Streaming**: Consola integrada en tiempo real que ejecuta comandos PowerShell (Windows) o sh (Unix) asíncronamente con control de detención (`kill`).
* **Ejecutor de Python**: Ejecución de scripts locales. Las gráficas y diagramas generados en `.png` se detectan y renderizan automáticamente en el chat.
* **Edición Inteligente**: Modificación de código mediante reescritura directa o parches incrementales (`patch_file`) con marcas `SEARCH/REPLACE` para ediciones seguras y rápidas.
* **Herramientas Personalizadas**: Panel para registrar tus propios comandos del sistema como herramientas del agente mediante plantillas (ej. `curl https://wttr.in/{{arg}}`).

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>Buscador de Código Local (RAG)
* **Algoritmo TF-IDF**: Motor de búsqueda indexado en Rust que recupera archivos relevantes en milisegundos.
* **Filtros Avanzados**: Soporte para segmentar búsquedas por extensiones de archivo (`ext:ts`) y subcarpetas (`path:utils`).

### <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z"/></svg>Live Code Editor & Integración de Código
* **Editor Integrado**: Visualización y edición en vivo de los archivos del proyecto con numeración de líneas sincronizada y guardado en disco directo.
* **Resaltado PrismJS**: Coloreado de sintaxis enriquecido con tema oscuro premium para JS, TS, Rust, Python, C++, Bash y más.
* **Aplicar al Editor**: Botón inteligente en las burbujas de chat para inyectar bloques de código generados por la IA directamente en el editor activo con un solo clic.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>Arquitectura y Tecnologías

* **Frontend**: React 18, TypeScript, Vite, CSS Vanilla (diseño responsive, glassmorphism y microanimaciones).
* **Coloreado**: PrismJS con tema *Tomorrow-Tomorrow*.
* **Backend**: Rust, Tauri v1 (control de subprocesos e integración de comandos nativos).
* **Base de Datos**: SQLite (mediante `rusqlite` en Rust) para persistir proyectos, chats, mensajes, prompts personalizados y herramientas.
* **API local**: Servidor compatible con la especificación de OpenAI corriendo en el puerto `1234`.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect width="20" height="12" x="2" y="3" rx="2"/><line x1="12" x2="12" y1="15" y2="21"/><line x1="8" x2="16" y1="21" y2="21"/></svg>Requisitos de Sistema

* **Sistema Operativo**: Windows (PowerShell habilitado) o Unix (Linux/macOS).
* **Node.js**: v18 o superior.
* **Manejador de paquetes**: `pnpm` instalado globalmente.
* **Rust**: Cadena de herramientas de Rust (`rustc`, `cargo` y `rustup`).
* **Ejecutables de Inferencia**: Binarios de inferencia locales (`llama-diffusion-cli.exe` o equivalentes de difusión/LLM) ubicados en el directorio `runtime/`.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 2v16M19 11l-7 7-7-7M2 22h20"/></svg>Instalación y Desarrollo

Sigue estos pasos para compilar y ejecutar Model Studio localmente en modo desarrollo:

1. **Clona el repositorio** e ingresa a la carpeta del proyecto.
2. **Instala las dependencias de Node.js**:
   ```powershell
   pnpm install
   ```
3. **Prepara el directorio Runtime**:
   Coloca tus ejecutables de inferencia y sus DLLs necesarias dentro de la carpeta `runtime/` en la raíz del proyecto.
4. **Configura tus Modelos**:
   Crea una carpeta `models/` y coloca tus archivos `.gguf` o checkpoints de difusión.
5. **Inicia la aplicación en modo desarrollo**:
   ```powershell
   pnpm tauri dev
   ```

Tauri compilará el backend en Rust, levantará el servidor de desarrollo de Vite y abrirá la ventana principal de Model Studio en tu escritorio.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.1a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>Configuración Avanzada

### Base de Datos SQLite
La aplicación inicializa y migra automáticamente una base de datos local donde gestiona:
* La tabla `projects` y `conversations` (con aislamiento de historial por proyecto).
* La tabla `agent_prompts` para almacenar las instrucciones personalizadas de tus agentes.
* Las `custom_tools` creadas desde la interfaz.

### Servidor API Local
Al iniciar la aplicación, se expone un servidor compatible con la API de OpenAI en `http://127.0.0.1:1234`. Puedes conectar otras herramientas locales (como extensiones de editores o wrappers de chat) directamente a este endpoint utilizando cualquier cliente estándar.

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>Licencia y Autoría

Desarrollado bajo estándares premium de experiencia de usuario y desarrollo de software por **DaosPath**. Todos los derechos reservados.
