import React from "react";

type Props = {
  disabled?: boolean;
  onPickFile: (file: File) => void;
};

function isPdf(file: File): boolean {
  const byType = file.type === "application/pdf";
  const byName = file.name.toLowerCase().endsWith(".pdf");
  return byType || byName;
}

export default function UploadZone({ disabled, onPickFile }: Props) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = React.useState(false);
  const [localError, setLocalError] = React.useState<string | null>(null);

  const pickFromDisk = () => inputRef.current?.click();

  const handleFiles = (files: FileList | null) => {
    setLocalError(null);
    const file = files?.[0];
    if (!file) return;
    if (!isPdf(file)) {
      setLocalError("Please select a PDF file.");
      return;
    }
    onPickFile(file);
  };

  const onDragOver: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragging(true);
  };

  const onDragLeave: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    handleFiles(e.dataTransfer.files);
  };

  const border = isDragging
    ? "1px solid rgba(122, 162, 255, 0.8)"
    : "1px dashed rgba(255, 255, 255, 0.22)";

  return (
    <div className="card">
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
        disabled={disabled}
      />

      <div
        onClick={disabled ? undefined : pickFromDisk}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === "Enter" || e.key === " ") pickFromDisk();
        }}
        style={{
          border,
          borderRadius: 14,
          padding: 18,
          cursor: disabled ? "not-allowed" : "pointer",
          background: isDragging ? "rgba(122, 162, 255, 0.10)" : "transparent",
          outline: "none"
        }}
      >
        <div style={{ fontWeight: 650, marginBottom: 6 }}>
          Drag & drop a PDF here, or click to choose.
        </div>
        <div className="hint">
          We’ll upload it to <code>/api/process</code> and render the returned
          interactive graph below.
        </div>

        {localError ? (
          <div style={{ marginTop: 10 }} className="error">
            {localError}
          </div>
        ) : null}
      </div>
    </div>
  );
}

