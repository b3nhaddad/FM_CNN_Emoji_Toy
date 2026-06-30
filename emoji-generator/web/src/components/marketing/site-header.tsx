import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 border-b border-[#171511]/10 bg-[#F7F4EC]/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
        <Link
          href="/"
          className="flex items-center gap-2 font-sans text-[1.05rem] font-black tracking-tight text-[#171511]"
        >
          <span
            aria-hidden
            className="flex size-7 items-center justify-center rounded-full border-[1.5px] border-[#171511] text-[0.65rem] font-mono font-bold"
          >
            EG
          </span>
          <span>
            Emoji<span className="text-[#E0481E]">Generator</span>
          </span>
        </Link>

        <nav className="flex items-center gap-4">
          <Link
            href="/terms"
            className="font-mono text-[0.72rem] font-medium uppercase tracking-[0.08em] text-[#171511]/70 transition-colors hover:text-[#171511]"
          >
            Terms
          </Link>
        </nav>
      </div>
    </header>
  );
}
