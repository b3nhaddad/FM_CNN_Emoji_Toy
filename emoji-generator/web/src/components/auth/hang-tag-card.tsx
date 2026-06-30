import type { ReactNode } from "react";

/**
 * Signature visual device for the auth flow: the auth "card" is rendered as a
 * literal garment hang-tag — a punched hole + loop at the top, a barcode-style
 * perforation strip standing in for dividers, sitting very slightly askew on
 * the paper backdrop like a tag resting on a tabletop. Depop is a secondhand
 * clothing marketplace, so the tag is the one artifact everyone in this
 * audience already recognizes.
 */
export function HangTagCard({
  eyebrow,
  title,
  subtitle,
  children,
  footer,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="relative mx-auto w-full max-w-[420px]">
      {/* loop string threading the punch hole, drawn behind the card */}
      <svg
        aria-hidden
        viewBox="0 0 60 40"
        className="pointer-events-none absolute -top-7 left-9 h-10 w-14 -rotate-[4deg] text-[#6B6358]/40"
      >
        <path
          d="M30 38 C 10 38, 8 4, 30 4 C 52 4, 50 38, 30 38"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
      </svg>

      <div
        className="relative -rotate-1 rounded-[4px] bg-[#FFFEFB] px-7 pt-9 pb-7 shadow-[0_1px_0_rgba(21,19,15,0.04),0_18px_40px_-12px_rgba(21,19,15,0.28)] ring-1 ring-[#15130F]/[0.06] transition-transform duration-300 ease-out focus-within:rotate-0 hover:rotate-0 sm:px-9 sm:pt-10"
        style={{
          backgroundImage:
            "radial-gradient(circle at 36px 26px, transparent 9px, rgba(21,19,15,0.08) 9px, rgba(21,19,15,0.08) 10px, transparent 10px)",
        }}
      >
        {/* punch hole */}
        <span className="absolute top-[26px] left-[28px] block size-[15px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#F4EFE6] ring-1 ring-inset ring-[#15130F]/15" />

        <p className="font-mono text-[11px] font-medium tracking-[0.18em] text-[#C81F40] uppercase">
          {eyebrow}
        </p>
        <h1 className="mt-2 font-[var(--font-syne)] text-[1.65rem] leading-tight font-extrabold tracking-tight text-[#15130F]">
          {title}
        </h1>
        {subtitle ? (
          <p className="mt-1.5 text-[0.92rem] text-[#6B6358]">{subtitle}</p>
        ) : null}

        <div className="mt-6">{children}</div>
      </div>

      {footer ? (
        <div className="mt-6 text-center text-sm text-[#6B6358]">{footer}</div>
      ) : null}
    </div>
  );
}

/** Barcode-style perforation divider standing in for "or" dividers. */
export function TagDivider({ label }: { label: string }) {
  const bars = [2, 1, 3, 1, 2, 1, 1, 3, 2, 1, 1, 2, 3, 1, 2, 1, 1, 2, 1, 3, 2, 1, 1, 2];
  return (
    <div className="my-5 flex items-center gap-3">
      <span className="flex h-3.5 flex-1 items-end gap-[2px] overflow-hidden opacity-30">
        {bars.map((w, i) => (
          <span
            key={i}
            className="block h-full bg-[#15130F]"
            style={{ width: `${w}px` }}
          />
        ))}
      </span>
      <span className="font-mono text-[10px] font-medium tracking-[0.18em] text-[#6B6358] uppercase">
        {label}
      </span>
      <span className="flex h-3.5 flex-1 items-end gap-[2px] overflow-hidden opacity-30">
        {bars
          .slice()
          .reverse()
          .map((w, i) => (
            <span
              key={i}
              className="block h-full bg-[#15130F]"
              style={{ width: `${w}px` }}
            />
          ))}
      </span>
    </div>
  );
}
