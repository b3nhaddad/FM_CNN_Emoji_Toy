/**
 * Signature hero visual: a swing price-tag rendered as a finished Depop
 * listing card, on a string, slightly rotated — the thing this product
 * actually produces. Pure CSS/SVG, no images required.
 */
export function ListingTag() {
  return (
    <div className="relative mx-auto w-[280px] select-none sm:w-[320px]">
      <style>{`
        @keyframes depop-tag-swing {
          0%, 100% { transform: rotate(-3deg); }
          50% { transform: rotate(3deg); }
        }
        .depop-tag-swing {
          transform-origin: top center;
          animation: depop-tag-swing 6s ease-in-out infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .depop-tag-swing { animation: none; }
        }
      `}</style>
      {/* string + punch hole */}
      <svg
        aria-hidden
        viewBox="0 0 320 90"
        className="absolute -top-[78px] left-1/2 w-[120px] -translate-x-1/2 overflow-visible"
      >
        <path
          d="M 10 0 Q 160 95 310 0"
          fill="none"
          stroke="#171511"
          strokeOpacity="0.35"
          strokeWidth="2"
        />
      </svg>

      <div className="depop-tag-swing">
        <div className="relative rounded-2xl border-[1.5px] border-[#171511] bg-[#FFFFFF] p-4 shadow-[6px_10px_0_0_rgba(23,21,17,0.12)]">
          {/* punch hole */}
          <div className="absolute left-1/2 top-3 size-3.5 -translate-x-1/2 rounded-full border-[1.5px] border-[#171511] bg-[#F7F4EC]" />

          <div className="mt-4 flex gap-3">
            <div className="flex size-20 shrink-0 items-center justify-center rounded-lg bg-[#F1ECE0] text-[1.6rem]">
              👕
            </div>
            <div className="flex flex-1 flex-col justify-center gap-1.5">
              <div className="h-2 w-full rounded-full bg-[#171511]/85" />
              <div className="h-2 w-4/5 rounded-full bg-[#171511]/30" />
              <div className="h-2 w-3/5 rounded-full bg-[#171511]/15" />
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between border-t border-dashed border-[#171511]/20 pt-3">
            <span className="font-mono text-[0.7rem] font-bold uppercase tracking-[0.06em] text-[#3B5D4F]">
              Ready to post
            </span>
            <span className="font-sans text-lg font-black text-[#E0481E]">
              $24
            </span>
          </div>
        </div>

        {/* little stamped badge, like a price gun sticker */}
        <div className="absolute -right-5 -top-3 flex size-12 -rotate-12 items-center justify-center rounded-full border-[1.5px] border-dashed border-[#E0481E] bg-[#F7F4EC] font-mono text-[0.6rem] font-bold uppercase leading-tight text-[#E0481E]">
          AI
          <br />
          gen
        </div>
      </div>
    </div>
  );
}
