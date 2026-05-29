import { useMemo } from "react";

interface Props {
  values: number[];
  width?: number;
  height?: number;
  className?: string;
  /** If true, color the stroke by trend sign (positive=green, negative=red). Otherwise signal cyan. */
  signed?: boolean;
}

// Minimal SVG sparkline. No deps. Reads cleanly at 40×14.
export function Sparkline({ values, width = 56, height = 14, className, signed = false }: Props) {
  const { d, sign } = useMemo(() => {
    if (!values || values.length < 2) return { d: "", sign: 0 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const stepX = width / (values.length - 1);
    const pts = values.map((v, i) => {
      const x = i * stepX;
      const y = height - ((v - min) / range) * (height - 2) - 1;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    });
    return { d: pts.join(" "), sign: values[values.length - 1]! - values[0]! };
  }, [values, width, height]);

  if (!d) {
    return (
      <svg width={width} height={height} className={className}>
        <line x1={0} x2={width} y1={height / 2} y2={height / 2} stroke="var(--color-border-strong)" strokeWidth={1} />
      </svg>
    );
  }
  const stroke = signed
    ? sign > 0
      ? "var(--color-positive)"
      : sign < 0
        ? "var(--color-negative)"
        : "var(--color-signal-dim)"
    : "var(--color-signal-dim)";

  return (
    <svg width={width} height={height} className={className} role="img" aria-label="sparkline">
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.25} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
