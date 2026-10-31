const API_BASE = '/api/v1';

export async function fetchHealth() {
  const res = await fetch('/health');
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchTenants() {
  const res = await fetch(`${API_BASE}/tenants`);
  if (!res.ok) throw new Error('Failed to fetch tenants');
  return res.json();
}

export async function createTenant(name) {
  const res = await fetch(`${API_BASE}/tenants`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  });
  if (!res.ok) throw new Error('Failed to create tenant');
  return res.json();
}

export async function fetchDocuments(tenantId) {
  const res = await fetch(`${API_BASE}/documents?tenant_id=${tenantId}`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function uploadDocument(tenantId, file) {
  const formData = new FormData();
  formData.append('tenant_id', tenantId);
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function deleteDocument(tenantId, documentId) {
  const res = await fetch(`${API_BASE}/documents/${documentId}?tenant_id=${tenantId}`, {
    method: 'DELETE'
  });
  if (!res.ok) throw new Error('Failed to delete document');
  return res.json();
}

export async function fetchSessions(tenantId) {
  const res = await fetch(`${API_BASE}/chat/sessions?tenant_id=${tenantId}`);
  if (!res.ok) throw new Error('Failed to fetch chat sessions');
  return res.json();
}

export async function fetchSessionMessages(tenantId, sessionId) {
  const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}/messages?tenant_id=${tenantId}`);
  if (!res.ok) throw new Error('Failed to fetch session messages');
  return res.json();
}

/**
 * Streams chat tokens and metadata using Server-Sent Events (SSE).
 */
export async function streamChatQuery({
  tenantId,
  query,
  sessionId,
  documentId,
  onEvent,
  onError,
  onDone
}) {
  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tenant_id: tenantId,
        query,
        session_id: sessionId,
        document_id: documentId
      })
    });

    if (!response.ok) {
      throw new Error(`Chat stream request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            onEvent(data);
            if (data.type === 'done') {
              onDone(data);
            }
          } catch (e) {
            console.error('Error parsing SSE frame:', e);
          }
        }
      }
    }
  } catch (err) {
    if (onError) onError(err);
  }
}
