import React from "react";

type Props = {
  graphHtml: string;
};

function injectBaseHref(html: string): string {
  // Helps relative URLs inside returned HTML resolve against this origin.
  // Safe to skip if the HTML is fully self-contained.
  if (/<base\s/i.test(html)) return html;
  const baseTag = `<base href="${window.location.origin}/">`;

  if (/<head[\s>]/i.test(html)) {
    return html.replace(/<head([\s>])/i, `<head$1\n${baseTag}\n`);
  }
  if (/<html[\s>]/i.test(html)) {
    return html.replace(/<html([\s>])/i, `<html$1><head>${baseTag}</head>`);
  }
  return `<!doctype html><html><head>${baseTag}</head><body>${html}</body></html>`;
}

export default function GraphViewer({ graphHtml }: Props) {
  const srcDoc = React.useMemo(() => injectBaseHref(graphHtml), [graphHtml]);

  const openInNewTab = () => {
    const blob = new Blob([graphHtml], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank", "noopener,noreferrer");
    // Note: we intentionally don't revoke immediately; the new tab may still read it.
  };

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ fontWeight: 650 }}>Graph</div>
        <div style={{ flex: 1 }} />
        <button className="btn" type="button" onClick={openInNewTab}>
          Open in new tab
        </button>
      </div>

      <div className="divider" />

      <div className="viewerShell">
        <iframe
          className="iframe"
          title="Interactive graph"
          srcDoc={srcDoc}
          // The graph HTML usually needs JS to run (vis/pyvis).
          sandbox="allow-scripts allow-same-origin"
          loading="lazy"
          referrerPolicy="no-referrer"
        />
      </div>
    </div>
  );
}

