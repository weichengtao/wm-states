import type { ComponentProps } from "react";
import { Input } from "./ui/input";

type Props = Omit<
  ComponentProps<"input">,
  "type" | "value" | "onChange" | "onBlur"
> & {
  value: string;
  integer?: boolean;
  error?: string;
  onDraft: (text: string) => void;
  onCommit: () => void;
};

/** Text inputs preserve partial decimals, signs, and exponents across keystrokes. */
export default function NumericInput({
  value,
  integer,
  error,
  onDraft,
  onCommit,
  ...props
}: Props) {
  const errorId = `${props.id}-error`;
  return (
    <>
      <Input
        {...props}
        type="text"
        inputMode={integer ? "numeric" : "decimal"}
        value={value}
        aria-invalid={!!error}
        aria-describedby={
          [props["aria-describedby"], error ? errorId : null]
            .filter(Boolean)
            .join(" ") || undefined
        }
        onChange={(event) => onDraft(event.target.value)}
        onBlur={onCommit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            onCommit();
          }
          props.onKeyDown?.(event);
        }}
      />
      {error && (
        <p id={errorId} className="text-destructive" role="alert">
          {error}
        </p>
      )}
    </>
  );
}
