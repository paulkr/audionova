/**
 * Flux field — full-bleed liquid gradient backdrop with a glass orb arcing
 * through the frame. Pure CSS/SVG, no client JS: renders on the server.
 *
 * Motion has three independent tempos so the field never visibly loops:
 *   1. Colour and ribbon bands drift INSIDE the warp, so the marbling folds
 *      and flows rather than sliding as one frozen sheet.
 *   2. The displacement scale breathes, pulling the whole warp in and out.
 *   3. Unwarped grading passes (tint, shade) rotate over the top.
 *
 * Cost note: 1 and 2 both re-run the turbulence on repaint, so the filter is
 * kept deliberately coarse — 3 octaves, a tight filter region, and a heavy
 * blur afterwards that hides the coarseness. Turn `--flux-motion` off (see
 * globals.css) to fall back to a static warp.
 */
export default function FluxField() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 overflow-hidden"
    >
      <svg className="absolute h-0 w-0" aria-hidden focusable="false">
        <defs>
          <filter
            id="flux-warp"
            x="-18%"
            y="-18%"
            width="136%"
            height="136%"
            filterUnits="objectBoundingBox"
            colorInterpolationFilters="sRGB"
          >
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.0016 0.0031"
              numOctaves={3}
              seed={11}
              result="noise"
            />
            <feDisplacementMap
              in="SourceGraphic"
              in2="noise"
              scale={520}
              xChannelSelector="R"
              yChannelSelector="G"
            >
              {/* The warp breathes — folds deepen and relax on a long cycle.
                  Prime-ish durations against the CSS tempos so the combined
                  motion doesn't settle into a visible repeat. */}
              <animate
                attributeName="scale"
                values="470;585;500;470"
                keyTimes="0;0.37;0.71;1"
                dur="53s"
                calcMode="spline"
                keySplines="0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1"
                repeatCount="indefinite"
              />
            </feDisplacementMap>
          </filter>
        </defs>
      </svg>

      <div className="absolute inset-0 bg-black" />

      {/* One warped stage. All five bands share a stacking context so their
          blend modes composite against each other — split across two filtered
          stages, the upper one isolates and hazes over the ribbon detail. */}
      <div className="flux flux--near">
        <div className="flux__band flux__band--a" />
        <div className="flux__band flux__band--b" />
        <div className="flux__band flux__band--ribbon" />
        <div className="flux__band flux__band--ribbon-fine" />
        <div className="flux__band flux__band--gleam" />
      </div>

      {/* Unwarped grading passes. These must be siblings, not children of a
          wrapper: a wrapper would form its own stacking context and their
          blend modes would never reach the warped stages underneath. */}
      <div className="flux-pass flux__band--tint" />
      <div className="flux-pass flux__band--shade" />

      <div className="orb">
        <div className="orb__sheen" />
        <div className="orb__terminator" />
      </div>

      <div className="vignette" />
      <div className="grain" />
    </div>
  );
}
