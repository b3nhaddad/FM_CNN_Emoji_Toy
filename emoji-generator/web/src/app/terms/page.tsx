import type { Metadata } from "next";
import Link from "next/link";
import { Eyebrow } from "@/components/marketing/eyebrow";
import { SiteHeader } from "@/components/marketing/site-header";
import { SiteFooter } from "@/components/marketing/site-footer";

export const metadata: Metadata = {
  title: "Terms of Use",
  description: "Terms and conditions for using the AI emoji generator.",
  alternates: { canonical: "/terms" },
  openGraph: {
    title: "Terms of Use | Emoji Generator",
    description: "Terms and conditions for using the AI emoji generator.",
    url: "/terms",
    type: "website",
  },
};

const SECTIONS = [
  {
    title: "No commercial use",
    body: "This model was trained on Apple's emoji dataset. Apple's emoji designs are copyrighted. Because the training data includes copyrighted material, any output from this model cannot be sold, licensed, or used in commercial products. Personal use only.",
  },
  {
    title: "No warranty",
    body: "This is a research project. The model is provided as-is with no guarantees about output quality, uptime, or accuracy. Generated emojis may be unexpected or low quality.",
  },
  {
    title: "Data & privacy",
    body: "Prompts you submit may be logged for debugging and model improvement. Do not submit sensitive personal information as a prompt.",
  },
  {
    title: "Fair use",
    body: "Please don't abuse the service with automated bulk generation. This is a free tool with limited compute — be reasonable.",
  },
];

export default function TermsPage() {
  return (
    <>
      <SiteHeader />

      <main className="flex-1">
        <section className="px-5 pb-12 pt-16 sm:px-8 sm:pt-24">
          <div className="mx-auto max-w-3xl text-center">
            <Eyebrow className="justify-center">Terms & Conditions</Eyebrow>
            <h1 className="mt-5 font-sans text-4xl font-black tracking-tight text-[#171511] sm:text-5xl">
              Yeah, you can&rsquo;t use this commercially.
            </h1>
            <p className="mx-auto mt-5 max-w-xl font-sans text-lg leading-relaxed text-[#171511]/60">
              The model was trained on Apple&rsquo;s emoji data. That comes with
              restrictions. Here&rsquo;s what you need to know.
            </p>
          </div>
        </section>

        <section className="px-5 pb-16 sm:px-8">
          <div className="mx-auto max-w-2xl space-y-8">
            {SECTIONS.map((section) => (
              <div
                key={section.title}
                className="rounded-2xl border border-[#171511]/10 bg-white/50 px-6 py-5"
              >
                <h2 className="font-mono text-[0.72rem] font-bold uppercase tracking-[0.12em] text-[#E0481E]">
                  {section.title}
                </h2>
                <p className="mt-2 font-sans text-base leading-relaxed text-[#171511]/70">
                  {section.body}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section className="px-5 pb-24 pt-4 sm:px-8">
          <div className="mx-auto max-w-3xl text-center">
            <Link
              href="/"
              className="font-mono text-[0.72rem] uppercase tracking-[0.08em] text-[#171511]/45 hover:text-[#171511]"
            >
              ← Back to generator
            </Link>
          </div>
        </section>
      </main>

      <SiteFooter />
    </>
  );
}
