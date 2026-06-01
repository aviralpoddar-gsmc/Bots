import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider leading-none",
  {
    variants: {
      variant: {
        default: "border-border text-text-2 bg-transparent",
        signal: "border-signal/40 text-signal bg-signal-bg/40",
        positive: "border-positive/30 text-positive bg-positive/5",
        negative: "border-negative/30 text-negative bg-negative/5",
        warn: "border-warn/30 text-warn bg-warn/5",
        solid: "border-text text-bg bg-text",
        ghost: "border-transparent text-text-3 bg-transparent",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
