import { useId, type ComponentPropsWithoutRef } from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

interface SelectProps extends Omit<
  ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>,
  "value" | "onChange" | "defaultValue" | "children"
> {
  value: string;
  onValueChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  name?: string;
}

// Radix reserves an empty value for its placeholder. Prefix every option so an
// explicit empty-string option (for nullable analysis settings) stays selectable.
const valuePrefix = "option:";

export function Select({
  value,
  onValueChange,
  options,
  placeholder = "Choose an option",
  disabled,
  name,
  className,
  ...props
}: SelectProps) {
  const descriptionId = useId();
  const selected = options.find((option) => option.value === value);
  const unavailable = disabled || !options.some((option) => !option.disabled);
  return (
    <SelectPrimitive.Root
      value={selected ? `${valuePrefix}${value}` : ""}
      onValueChange={(next) => onValueChange(next.slice(valuePrefix.length))}
      disabled={unavailable}
      name={name}
    >
      <SelectPrimitive.Trigger
        className={cn("select-control ui-select-trigger", className)}
        title={
          selected
            ? [selected.label, selected.description].filter(Boolean).join(" · ")
            : placeholder
        }
        {...props}
      >
        <span className="ui-select-value">
          <SelectPrimitive.Value placeholder={placeholder}>
            {selected?.label}
          </SelectPrimitive.Value>
        </span>
        <SelectPrimitive.Icon className="ui-select-chevron">
          <ChevronDown size={16} aria-hidden="true" />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          className="ui-select-content"
          data-descriptions={
            options.some((option) => option.description) || undefined
          }
          position="popper"
          sideOffset={7}
          collisionPadding={12}
          align="start"
        >
          <SelectPrimitive.ScrollUpButton className="ui-select-scroll">
            <ChevronUp size={16} aria-hidden="true" />
          </SelectPrimitive.ScrollUpButton>
          <SelectPrimitive.Viewport className="ui-select-viewport">
            {options.map((option, index) => (
              <SelectPrimitive.Item
                key={option.value}
                value={`${valuePrefix}${option.value}`}
                disabled={option.disabled}
                textValue={option.label}
                aria-describedby={
                  option.description ? `${descriptionId}-${index}` : undefined
                }
                className="ui-select-item"
              >
                <span className="ui-select-option-copy">
                  <SelectPrimitive.ItemText>
                    {option.label}
                  </SelectPrimitive.ItemText>
                  {option.description && (
                    <span
                      id={`${descriptionId}-${index}`}
                      className="ui-select-description"
                    >
                      {option.description}
                    </span>
                  )}
                </span>
                <SelectPrimitive.ItemIndicator className="ui-select-check">
                  <Check size={16} strokeWidth={2.5} aria-hidden="true" />
                </SelectPrimitive.ItemIndicator>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
          <SelectPrimitive.ScrollDownButton className="ui-select-scroll">
            <ChevronDown size={16} aria-hidden="true" />
          </SelectPrimitive.ScrollDownButton>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}
