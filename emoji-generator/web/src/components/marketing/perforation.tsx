import { cn } from "@/lib/utils";

/**
 * A tear-off perforation line — the recurring section-break motif across the
 * marketing site, styled after a receipt / garment-tag tear strip.
 */
export function Perforation({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn("mx-auto max-w-6xl px-5 sm:px-8", className)}
    >
      <div className="relative flex items-center">
        <span className="-ml-2.5 size-5 rounded-full border border-[#171511]/15 bg-[#F7F4EC]" />
        <div className="h-0 flex-1 border-t border-dashed border-[#171511]/20" />
        <span className="-mr-2.5 size-5 rounded-full border border-[#171511]/15 bg-[#F7F4EC]" />
      </div>
    </div>
  );
}
