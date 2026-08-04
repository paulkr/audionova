import FluxField from "../flux-field";
import UploadPanel from "../upload-panel";

const NAV = ["PROCESS", "MODELS", "STUDIO", "PRICING", "CONTACT"];
const FORMATS = ["WAV", "MP3", "FLAC", "M4A"];

export default function Home() {
  return (
    <div className="relative h-full w-full overflow-hidden bg-canvas">
      <FluxField />

      {/* ---- Chrome: transparent bar, logo left / switcher centre / nav right ---- */}
      <header className="absolute inset-x-0 top-0 z-10 flex items-start justify-between px-8 py-8 sm:px-12 sm:py-10">
        <a
          href="#"
          className="text-[16px] leading-none tracking-[-0.01em] text-frost"
        >
          <span className="font-semibold">audio</span>
          <span className="font-light text-white/70">nova</span>
        </a>

        <nav className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-5 pt-1 md:flex">
          {FORMATS.map((f, i) => (
            <span
              key={f}
              className={`text-[11px] uppercase leading-[1.58] tracking-[0.06em] ${
                i === 0 ? "text-frost" : "text-white/45"
              }`}
            >
              {f}
            </span>
          ))}
        </nav>

        <nav className="flex flex-col items-end gap-[3px]">
          {NAV.map((item) => (
            <a key={item} href="#" className="nav-link">
              {item}
            </a>
          ))}
        </nav>
      </header>

      {/* ---- Hero: centred headline over the media, single gesture ---- */}
      <main className="relative z-10 mx-auto flex h-full w-full max-w-[1078px] flex-col items-center justify-center px-8 text-center">
        <h1
          className="rise max-w-[16ch] text-[clamp(2.75rem,7.4vw,5.25rem)] font-normal leading-[1.1] tracking-[-0.015em] text-frost"
          style={{ animationDelay: "0.1s" }}
        >
          Sound, Unbound
        </h1>

        <p
          className="rise mt-6 max-w-[46ch] text-[18px] font-light leading-[1.22] text-white/60"
          style={{ animationDelay: "0.25s" }}
        >
          Drop a track. Say how you want it to sound. Audionova shapes it in a
          single pass.
        </p>

        <div
          className="rise mt-[46px] flex w-full justify-center"
          style={{ animationDelay: "0.4s" }}
        >
          <UploadPanel />
        </div>
      </main>

      {/* ---- Rotating badge, bottom-left ---- */}
      <div className="absolute bottom-10 left-8 z-10 hidden h-[76px] w-[76px] items-center justify-center sm:left-12 sm:flex">
        <svg
          viewBox="0 0 100 100"
          className="badge-rotate absolute inset-0 h-full w-full"
          aria-hidden
        >
          <defs>
            <path
              id="badge-arc"
              d="M50,50 m-37,0 a37,37 0 1,1 74,0 a37,37 0 1,1 -74,0"
              fill="none"
            />
          </defs>
          <text
            fill="rgba(255,255,255,0.72)"
            fontSize="9"
            fontFamily="var(--font-figtree), system-ui, sans-serif"
          >
            {/* textLength = circumference of r=37, so one pass closes the loop exactly */}
            <textPath
              href="#badge-arc"
              startOffset="0"
              textLength={232.5}
              lengthAdjust="spacing"
            >
              DROP · YOUR · AUDIO ·
            </textPath>
          </text>
        </svg>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
          <circle
            cx="8"
            cy="8"
            r="5.4"
            stroke="rgba(255,255,255,0.8)"
            strokeWidth="1"
          />
          <path
            d="M8 5.4v5.2M5.6 8h4.8"
            stroke="rgba(255,255,255,0.8)"
            strokeWidth="1"
            strokeLinecap="round"
          />
        </svg>
      </div>

      {/* ---- Utility chip, bottom-right ---- */}
      <div className="absolute bottom-10 right-8 z-10 hidden items-center gap-3 sm:right-12 sm:flex">
        <span className="text-[11px] uppercase leading-[1.58] tracking-[0.06em] text-white/45">
          Engine v2 · 48 kHz
        </span>
        <button
          type="button"
          className="pill-ghost px-[18px] py-[7px] text-[11px] uppercase leading-[1.58] tracking-[0.06em]"
        >
          Docs
        </button>
      </div>
    </div>
  );
}
