import axios from 'axios'

const api = axios.create({ baseURL: import.meta.env.VITE_API_URL })

export const postQuery = (question) => api.post('/query', { question }).then((r) => r.data)

export const getMetrics = () => api.get('/metrics').then((r) => r.data)

export const postFeedback = (query_id, thumbs_up) =>
  api.post('/feedback', { query_id, thumbs_up }).then((r) => r.data)
