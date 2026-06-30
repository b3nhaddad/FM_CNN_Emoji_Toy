import Link from "next/link";

const COLUMNS = [
  {
    heading: "Legal",
    links: [
      { href: "/terms", label: "Terms of use" },
    ],
  },
  {
    heading: "Me",
    links: [
      { href: "https://github.com/b3nhaddad", label: "Github" },
      { href: "https://www.linkedin.com/in/benjamin-steven-haddad/", label: "LinkedIn" },
    ],
  },
];

export function SiteFooter() {
  return (
    <footer className="border-t border-[#171511]/10 bg-[#F7F4EC]">
      <div className="mx-auto max-w-6xl px-5 py-14 sm:px-8">
        <div className="grid grid-cols-2 gap-10 sm:grid-cols-4">
          <div className="col-span-2">
            <Link
              href="/"
              className="font-sans text-lg font-black tracking-tight text-[#171511]"
            >
              Emoji<span className="text-[#E0481E]">Generator</span>
            </Link>
            <p className="mt-3 max-w-xs font-sans text-sm leading-relaxed text-[#171511]/60">
              AI-powered emoji generation using flow matching. For personal use only.
            </p>
          </div>
          {COLUMNS.map((col) => (
            <div key={col.heading}>
              <p className="font-mono text-[0.68rem] font-bold uppercase tracking-[0.1em] text-[#171511]/45">
                {col.heading}
              </p>
              <ul className="mt-4 flex flex-col gap-2.5">
                {col.links.map((link) => (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      className="font-sans text-sm text-[#171511]/70 transition-colors hover:text-[#E0481E]"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col gap-3 border-t border-dashed border-[#171511]/20 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="font-mono text-[0.7rem] text-[#171511]/45">
            Not affiliated with Apple
          </p>
        </div>
      </div>
    </footer>
  );
}
