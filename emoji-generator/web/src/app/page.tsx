"use client";

import { useState } from "react";
import { SiteHeader } from "@/components/marketing/site-header";
import { SiteFooter } from "@/components/marketing/site-footer";
import { Eyebrow } from "@/components/marketing/eyebrow";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Download, RefreshCw, Loader2 } from "lucide-react";

type GeneratorState = "idle" | "loading" | "result" | "error";

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [state, setState] = useState<GeneratorState>("idle");
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleGenerate() {
    if (!prompt.trim()) return;
    setState("loading");
    setError(null);
    setResultUrl(null);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim() }),
      });

      if (!res.ok) {
        throw new Error(`Generation failed: ${res.statusText}`);
      }

      const data = await res.json() as { url: string };
      setResultUrl(data.url);
      setState("result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setState("error");
    }
  }

  function handleRedo() {
    setState("idle");
    setResultUrl(null);
    setError(null);
  }

  return (
    <>
      <SiteHeader />

      <main className="flex-1">
        <section className="px-5 pb-12 pt-16 sm:px-8 sm:pt-24">
          <div className="mx-auto max-w-2xl text-center">
            <Eyebrow className="justify-center">Flow Matching Model</Eyebrow>
            <h1 className="mt-5 font-sans text-4xl font-black tracking-tight text-[#171511] sm:text-5xl">
              Generate your emoji
            </h1>
            <p className="mx-auto mt-5 max-w-lg font-sans text-lg leading-relaxed text-[#171511]/60">
              Describe an emoji and our AI model will generate it. For personal
              use only — see{" "}
              <a href="/terms" className="underline underline-offset-4 hover:text-[#E0481E]">
                terms
              </a>{" "}
              for details.
            </p>
          </div>
        </section>

        <section className="px-5 pb-24 sm:px-8">
          <div className="mx-auto max-w-xl">
            {state !== "result" && (
              <div className="space-y-4">
                <Textarea
                  placeholder="A smiling sun wearing sunglasses..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="min-h-28 resize-none border-[#171511]/20 bg-white/60 font-sans text-base placeholder:text-[#171511]/30 focus-visible:border-[#E0481E] focus-visible:ring-[#E0481E]/20"
                  disabled={state === "loading"}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                      void handleGenerate();
                    }
                  }}
                />

                {state === "loading" && (
                  <div className="rounded-2xl border border-[#171511]/10 bg-white/40 p-8 text-center">
                    <div className="flex flex-col items-center gap-3">
                      <Loader2 className="size-8 animate-spin text-[#E0481E]" />
                      <p className="font-mono text-sm text-[#171511]/60">
                        Generating your emoji…
                      </p>
                    </div>
                  </div>
                )}

                {state === "error" && (
                  <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-center">
                    <p className="font-sans text-sm text-red-700">{error}</p>
                  </div>
                )}

                <Button
                  onClick={() => void handleGenerate()}
                  disabled={!prompt.trim() || state === "loading"}
                  className="w-full rounded-full bg-[#171511] py-3 font-mono text-[0.8rem] font-bold uppercase tracking-[0.08em] text-[#F7F4EC] transition-transform hover:-translate-y-0.5 disabled:opacity-40"
                >
                  {state === "loading" ? "Generating…" : "Generate Emoji"}
                </Button>

                <p className="text-center font-mono text-[0.65rem] text-[#171511]/40">
                  ⌘ + Enter to generate
                </p>
              </div>
            )}

            {state === "result" && resultUrl && (
              <div className="flex flex-col items-center gap-6">
                <div className="overflow-hidden rounded-3xl border border-[#171511]/10 bg-white p-6 shadow-sm">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={resultUrl}
                    alt={`Generated emoji: ${prompt}`}
                    className="size-48 object-contain"
                  />
                </div>

                <p className="max-w-sm text-center font-sans text-sm text-[#171511]/60">
                  &ldquo;{prompt}&rdquo;
                </p>

                <div className="flex gap-3">
                  <a
                    href={resultUrl}
                    download="emoji.png"
                    className="inline-flex items-center gap-2 rounded-full bg-[#171511] px-5 py-2.5 font-mono text-[0.72rem] font-bold uppercase tracking-[0.08em] text-[#F7F4EC] transition-transform hover:-translate-y-0.5"
                  >
                    <Download className="size-3.5" />
                    Download
                  </a>
                  <Button
                    onClick={handleRedo}
                    variant="outline"
                    className="inline-flex items-center gap-2 rounded-full border-[#171511]/20 px-5 py-2.5 font-mono text-[0.72rem] font-bold uppercase tracking-[0.08em] text-[#171511] transition-transform hover:-translate-y-0.5"
                  >
                    <RefreshCw className="size-3.5" />
                    Try again
                  </Button>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      <SiteFooter />
    </>
  );
}
