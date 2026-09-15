const BASE = import.meta.env.VITE_API_BASE || ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}/api${path}`, {
    headers: options.body instanceof FormData
      ? undefined
      : { 'Content-Type': 'application/json' },
    ...options,
  })
  if (res.status === 204) return null
  const text = await res.text()
  let data
  try { data = text ? JSON.parse(text) : null } catch { data = { detail: text } }
  if (!res.ok) {
    throw new Error(data?.detail || `Request failed (${res.status}).`)
  }
  return data
}

export const api = {
  health: () => request('/health'),
  meta: () => request('/meta'),
  seedDemo: () => request('/demo/seed', { method: 'POST' }),

  listClients: () => request('/clients'),
  createClient: (payload) =>
    request('/clients', { method: 'POST', body: JSON.stringify(payload) }),
  deleteClient: (id) => request(`/clients/${id}`, { method: 'DELETE' }),

  trajectory: (id) => request(`/clients/${id}/trajectory`),
  therapistImpact: (id) => request(`/clients/${id}/therapist-impact`),
  insights: (id) => request(`/clients/${id}/insights`),
  profile: (id) => request(`/clients/${id}/profile`),
  models: () => request('/models'),
  uploadMedia: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return request('/upload/media', { method: 'POST', body: fd })
  },
  multimodal: (payload) =>
    request('/sessions/multimodal', { method: 'POST', body: JSON.stringify(payload) }),

  addSession: (id, payload) =>
    request(`/clients/${id}/sessions`, { method: 'POST', body: JSON.stringify(payload) }),
  uploadSession: (id, file) => {
    const fd = new FormData()
    fd.append('file', file)
    return request(`/clients/${id}/sessions/upload`, { method: 'POST', body: fd })
  },
  session: (sid) => request(`/sessions/${sid}`),
  deleteSession: (sid) => request(`/sessions/${sid}`, { method: 'DELETE' }),

  addMeasure: (id, payload) =>
    request(`/clients/${id}/measures`, { method: 'POST', body: JSON.stringify(payload) }),
  calibration: () => request('/calibration/weights'),

  analyze: (transcript) =>
    request('/analyze', { method: 'POST', body: JSON.stringify({ transcript }) }),
}

export const DIMENSION_LABELS = {
  self_agency: 'Self-agency',
  future_orientation: 'Future orientation',
  emotional_granularity: 'Emotional granularity',
  problem_ownership: 'Problem ownership',
  reflection_depth: 'Reflection depth',
}

export const STATE_COPY = {
  gaining: 'Gaining momentum',
  regressing: 'Losing ground',
  plateauing: 'Plateau',
  steady: 'Mixed / steady',
  insufficient_data: 'Not enough sessions',
}

export function stateColor(state) {
  if (state === 'gaining') return 'var(--gain)'
  if (state === 'regressing') return 'var(--regress)'
  return 'var(--hold)'
}
