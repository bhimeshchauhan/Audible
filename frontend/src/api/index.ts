/**
 * API client for the PDF Character Graph backend
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export interface ProcessResponse {
  graphHtml: string;
}

export interface ApiError {
  error: string;
  detail?: string;
}

/**
 * Maximum file size allowed for upload (50MB)
 */
export const MAX_FILE_SIZE = 50 * 1024 * 1024;

/**
 * Upload a PDF file and get the character relationship graph HTML
 */
export async function processPdf(file: File): Promise<ProcessResponse> {
  // Client-side validation
  if (file.size > MAX_FILE_SIZE) {
    throw new Error(`File too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB.`);
  }

  if (!file.name.toLowerCase().endsWith('.pdf')) {
    throw new Error('Please select a PDF file.');
  }

  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/api/process`, {
    method: 'POST',
    body: form,
  });

  const contentType = res.headers.get('content-type') ?? '';
  const isJson = contentType.toLowerCase().includes('application/json');

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      if (isJson) {
        const body = (await res.json()) as ApiError;
        detail = body.error || body.detail || detail;
      } else {
        const text = await res.text();
        detail = text?.slice(0, 500) || detail;
      }
    } catch {
      // ignore parsing errors; keep fallback message
    }
    throw new Error(detail);
  }

  if (!isJson) {
    throw new Error('Expected JSON response with { graphHtml }.');
  }

  const data = (await res.json()) as { graphHtml?: unknown };
  if (typeof data.graphHtml !== 'string' || data.graphHtml.trim() === '') {
    throw new Error('Response JSON did not include a non-empty graphHtml string.');
  }

  return { graphHtml: data.graphHtml };
}

/**
 * Health check endpoint
 */
export async function healthCheck(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error('Backend health check failed');
  }
  return res.json();
}
