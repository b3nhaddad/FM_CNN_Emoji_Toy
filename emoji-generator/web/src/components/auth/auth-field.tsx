import type { InputHTMLAttributes } from "react";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

export function AuthField({
  id,
  label,
  className,
  ...props
}: { id: string; label: string } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label
        htmlFor={id}
        className="text-[11px] font-semibold tracking-[0.1em] text-[#6B6358] uppercase"
      >
        {label}
      </Label>
      <input
        id={id}
        name={id}
        data-slot="input"
        className={cn(
          "h-11 w-full min-w-0 rounded-[6px] border border-[#15130F]/15 bg-[#FBF9F3] px-3.5 text-[0.95rem] text-[#15130F] outline-none transition-colors placeholder:text-[#6B6358]/60 focus-visible:border-[#FF2D55] focus-visible:bg-white focus-visible:ring-3 focus-visible:ring-[#FF2D55]/15 disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
        {...props}
      />
    </div>
  );
}
