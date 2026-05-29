import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-xs font-medium font-mono uppercase tracking-wider transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-signal disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-surface-2 text-text border border-border hover:border-border-strong hover:bg-surface-3",
        signal:
          "bg-signal-bg text-signal border border-signal/40 hover:bg-signal/15 hover:border-signal",
        ghost:
          "bg-transparent text-text-2 border border-transparent hover:bg-surface-2 hover:text-text",
        danger:
          "bg-transparent text-negative border border-negative/40 hover:bg-negative/10",
      },
      size: {
        default: "h-7 px-3",
        sm: "h-6 px-2 text-[10px]",
        icon: "h-7 w-7",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
    );
  },
);
Button.displayName = "Button";

export { buttonVariants };
