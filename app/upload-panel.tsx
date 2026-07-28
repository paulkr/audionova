"use client";

import { useRef, useState } from "react";

export default function UploadPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);

  function pick(list: FileList | null) {
    const next = list?.[0];
    if (next) setFile(next);
  }

  return (
    <div className="flex w-full max-w-[560px] flex-col items-center gap-[14px]">
      <div
        data-dragging={dragging}
        className="drop-well flex w-full items-center gap-4 px-5 py-4 sm:px-6"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files);
        }}
      >
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex flex-1 items-center gap-3 text-left"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 18 18"
            fill="none"
            aria-hidden
            className="shrink-0 opacity-70"
          >
            <path
              d="M2 11.5v3a1.5 1.5 0 0 0 1.5 1.5h11a1.5 1.5 0 0 0 1.5-1.5v-3M9 12V2m0 0L5.5 5.5M9 2l3.5 3.5"
              stroke="#fff"
              strokeWidth="1.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span
            className={`truncate text-[16px] leading-[1.25] ${
              file ? "text-white" : "text-white/55"
            }`}
          >
            {file ? file.name : "Drop an audio file, or browse"}
          </span>
        </button>

        <span className="hidden shrink-0 text-[11px] uppercase leading-[1.58] tracking-[0.06em] text-white/40 sm:block">
          {file ? `${(file.size / 1_048_576).toFixed(1)} MB` : "WAV · MP3 · FLAC"}
        </span>

        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          className="sr-only"
          onChange={(e) => pick(e.target.files)}
        />
      </div>

      <button
        type="button"
        disabled={!file}
        className="pill-filled px-[33.7px] py-[11.2px] text-[16px] leading-[1.25]"
      >
        Process
      </button>
    </div>
  );
}
