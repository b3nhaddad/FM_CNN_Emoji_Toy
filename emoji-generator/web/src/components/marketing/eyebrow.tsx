import { cn } from "@/lib/utils";

/**
 * Section label styled like a stamped price-tag code — Geist Mono,
 * uppercase, wide tracking. Used in place of generic "eyebrow" pill badges.
 */
export function Eyebrow({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <p
      className={cn(
        "inline-flex items-center gap-2 font-mono text-[0.72rem] font-bold uppercase tracking-[0.18em] text-[#E0481E]",
        className
      )}
    >
      <span aria-hidden className="inline-block size-1.5 rounded-full bg-[#E0481E]" />
      {children}
    </p>
  );
}
