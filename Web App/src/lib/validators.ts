import type { ErrorCode, ValidationResult } from '../types'

export const MAX_UPLOAD_MB = 10
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
export const ACCEPTED_FILE_TYPES = '.jpg,.jpeg,.png,.webp'

const MAX_URL_EXTRACTION_DEPTH = 4
const MAX_CANDIDATE_COUNT = 48
const ACCEPTED_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp'])
const WRAPPER_PARAM_KEYS = new Set([
  'mediaurl', 'imgurl', 'imageurl', 'image_url', 'image', 'img', 'url', 'u', 'r',
  'redirect', 'redirect_url', 'redirecturl', 'target', 'destination', 'dest',
  'src', 'source', 'media', 'photo', 'picture', 'original',
])
const IMAGE_EXTENSION_PATTERN = /\.(avif|bmp|gif|heic|heif|jfif|jpe?g|png|svg|webp)(?:$|[/?#])/i
const IMAGE_FORMAT_HINTS = ['avif', 'bmp', 'gif', 'heic', 'heif', 'jfif', 'jpg', 'jpeg', 'png', 'svg', 'webp']
const HTTP_URL_IN_TEXT_PATTERN = /https?:\/\/[^\s"'<>]+/gi
const WWW_URL_IN_TEXT_PATTERN = /\bwww\.[^\s"'<>]+/gi
const MARKDOWN_LINK_PATTERN = /\[[^[\]]*]\(([^)]+)\)/gi
const HTML_ATTR_URL_PATTERN = /\b(?:src|href|content|data-src|data-original|data-image)\s*=\s*["']([^"']+)["']/gi
const CSS_URL_PATTERN = /url\(([^)]+)\)/gi
const JSON_URL_FIELD_PATTERN = /"(?:url|src|image|image_url|imageUrl|thumbnail|thumbnailUrl)"\s*:\s*"([^"]+)"/gi
const SRCSET_VALUE_PATTERN = /\bsrcset\s*=\s*["']([^"']+)["']/gi
const EMBEDDED_URL_PATTERN = /(?:https?:\/\/|www\.|https?%3A%2F%2F)/i

const ERROR_MESSAGES: Record<ErrorCode, string> = {
  invalid_type: 'Please upload a JPG, PNG, or WEBP image.',
  file_too_large: 'This image is larger than 10 MB. Choose a smaller file.',
  invalid_url: 'Enter a valid public image URL.',
  cors_blocked: 'This website does not permit direct image access. Download the image and upload it instead.',
  image_load_failed: 'The image could not be loaded. Try another image or upload it directly.',
  classification_failed: 'The prediction could not be completed. Please try again.',
}

export const createValidationError = (errorCode: ErrorCode): ValidationResult => ({
  ok: false,
  errorCode,
  message: ERROR_MESSAGES[errorCode],
})

const collectPatternMatches = (source: string, pattern: RegExp, groupIndex = 0): string[] => {
  const matches: string[] = []
  const matcher = new RegExp(pattern.source, pattern.flags)
  let match: RegExpExecArray | null = matcher.exec(source)
  while (match) {
    const value = match[groupIndex] ?? match[0]
    if (value) matches.push(value)
    match = matcher.exec(source)
  }
  return matches
}

const decodeHtmlEntities = (value: string): string => value
  .replace(/&amp;/gi, '&')
  .replace(/&quot;/gi, '"')
  .replace(/&#39;/gi, "'")
  .replace(/&lt;/gi, '<')
  .replace(/&gt;/gi, '>')

const sanitizeInputValue = (value: string): string => decodeHtmlEntities(value)
  .replace(/\\u0026/gi, '&')
  .replace(/\\u003d/gi, '=')
  .replace(/\\u002f/gi, '/')
  .replace(/\\\//g, '/')
  .trim()
  .replace(/^["'`]+|["'`]+$/g, '')

const normalizeCandidateToken = (value: string): string => sanitizeInputValue(value)
  .replace(/^["'`([{<\s]+/, '')
  .replace(/[>"'`)\]}.,;!?\s]+$/, '')

const extractInlineUrlCandidates = (value: string): string[] => {
  const source = sanitizeInputValue(value)
  if (!source) return []
  const candidates = new Set<string>()
  const push = (candidate: string) => {
    if (candidates.size >= MAX_CANDIDATE_COUNT) return
    const normalized = normalizeCandidateToken(candidate)
    if (normalized) candidates.add(normalized)
  }

  push(source)
  collectPatternMatches(source, HTML_ATTR_URL_PATTERN, 1).forEach(push)
  collectPatternMatches(source, CSS_URL_PATTERN, 1).forEach(push)
  collectPatternMatches(source, JSON_URL_FIELD_PATTERN, 1).forEach(push)
  collectPatternMatches(source, MARKDOWN_LINK_PATTERN, 1).forEach((candidate) => push(candidate.trim().split(/\s+/)[0] ?? candidate))
  collectPatternMatches(source, SRCSET_VALUE_PATTERN, 1).forEach((srcset) => {
    srcset.split(',').forEach((entry) => push(entry.trim().split(/\s+/)[0] ?? ''))
  })
  collectPatternMatches(source, HTTP_URL_IN_TEXT_PATTERN).forEach(push)
  collectPatternMatches(source, WWW_URL_IN_TEXT_PATTERN).forEach(push)
  return Array.from(candidates)
}

const ensureProtocol = (value: string): string => {
  if (value.startsWith('//')) return `https:${value}`
  if (/^[a-zA-Z][a-zA-Z\d+.-]*:/.test(value)) return value
  if (/^www\./i.test(value)) return `https://${value}`
  if (/^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:[/:?#]|$)/.test(value)) return `https://${value}`
  return value
}

const parseHttpUrl = (value: string): URL | null => {
  for (const candidate of extractInlineUrlCandidates(value)) {
    try {
      const parsed = new URL(ensureProtocol(candidate))
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') return parsed
    } catch {
      // Continue through the extracted candidates.
    }
  }
  return null
}

const buildParamCandidates = (value: string): string[] => {
  const queue = [value]
  const candidates = new Set<string>()
  while (queue.length && candidates.size < MAX_CANDIDATE_COUNT) {
    const current = normalizeCandidateToken(queue.shift() ?? '')
    if (!current || candidates.has(current)) continue
    candidates.add(current)
    for (const decodeInput of [current, current.replace(/\+/g, '%20')]) {
      try {
        const decoded = normalizeCandidateToken(decodeURIComponent(decodeInput))
        if (decoded && !candidates.has(decoded)) queue.push(decoded)
      } catch {
        // Ignore malformed encoded values.
      }
    }
  }
  return Array.from(candidates)
}

const looksLikeImageUrl = (url: URL): boolean => {
  if (IMAGE_EXTENSION_PATTERN.test(url.pathname)) return true
  return ['fm', 'format', 'ext', 'mime', 'type'].some((key) => {
    const value = url.searchParams.get(key)?.toLowerCase()
    return value ? IMAGE_FORMAT_HINTS.some((hint) => value.includes(hint)) : false
  })
}

const normalizeKnownImageHostUrl = (url: URL): URL => {
  const hostname = url.hostname.toLowerCase()
  const isWikiHost = hostname === 'commons.wikimedia.org' || hostname.endsWith('.wikimedia.org') || hostname.endsWith('.wikipedia.org')
  if (!isWikiHost) return url
  const pathname = decodeURIComponent(url.pathname)
  const title = (pathname.match(/^\/wiki\/Special:FilePath\/(.+)$/i)?.[1] ?? pathname.match(/^\/wiki\/File:(.+)$/i)?.[1] ?? '').trim()
  if (!title) return url
  const normalizedTitle = encodeURIComponent(title.replace(/\s+/g, '_')).replace(/%2F/gi, '/')
  return new URL(`/wiki/Special:FilePath/${normalizedTitle}`, url.origin)
}

const collectSearchLikeParams = (url: URL): URLSearchParams[] => {
  const params = [url.searchParams]
  const hash = url.hash.replace(/^#\??/, '')
  if (hash && /[=&]/.test(hash)) params.push(new URLSearchParams(hash))
  return params
}

const collectNestedCandidates = (url: URL): string[] => {
  const prioritized: string[] = []
  const fallback: string[] = []
  const seen = new Set<string>()
  const push = (bucket: string[], value: string) => {
    for (const candidate of buildParamCandidates(value)) {
      if (seen.size >= MAX_CANDIDATE_COUNT) return
      if (!seen.has(candidate)) {
        seen.add(candidate)
        bucket.push(candidate)
      }
    }
  }
  for (const params of collectSearchLikeParams(url)) {
    for (const [rawKey, rawValue] of params.entries()) {
      if (WRAPPER_PARAM_KEYS.has(rawKey.toLowerCase())) push(prioritized, rawValue)
      else if (EMBEDDED_URL_PATTERN.test(rawValue) || IMAGE_EXTENSION_PATTERN.test(rawValue)) push(fallback, rawValue)
    }
  }
  extractInlineUrlCandidates(url.hash).forEach((candidate) => push(fallback, candidate))
  extractInlineUrlCandidates(url.pathname).forEach((candidate) => push(fallback, candidate))
  return [...prioritized, ...fallback]
}

const resolveImageUrl = (candidate: string, depth: number, visited: Set<string>): { url: URL | null; extracted: boolean } => {
  if (depth > MAX_URL_EXTRACTION_DEPTH) return { url: null, extracted: false }
  const parsed = parseHttpUrl(candidate)
  if (!parsed) return { url: null, extracted: false }
  const canonical = parsed.toString()
  if (visited.has(canonical)) return { url: parsed, extracted: false }
  visited.add(canonical)
  for (const nested of collectNestedCandidates(parsed)) {
    const resolved = resolveImageUrl(nested, depth + 1, visited)
    if (!resolved.url || resolved.url.toString() === canonical) continue
    if (resolved.url.hostname !== parsed.hostname || looksLikeImageUrl(resolved.url) || resolved.extracted) {
      return { url: resolved.url, extracted: true }
    }
  }
  return { url: parsed, extracted: false }
}

export const prepareImageUrl = (rawInput: string): { normalizedUrl: string | null; validation: ValidationResult } => {
  const cleaned = sanitizeInputValue(rawInput)
  if (!cleaned) return { normalizedUrl: null, validation: createValidationError('invalid_url') }
  const inputCandidates = new Set<string>([cleaned, ...extractInlineUrlCandidates(cleaned)])
  for (const candidate of Array.from(inputCandidates)) buildParamCandidates(candidate).forEach((item) => inputCandidates.add(item))
  let fallback: URL | null = null
  for (const candidate of inputCandidates) {
    const resolved = resolveImageUrl(candidate, 0, new Set<string>())
    if (!resolved.url) continue
    const normalized = normalizeKnownImageHostUrl(resolved.url)
    if (looksLikeImageUrl(normalized)) return { normalizedUrl: normalized.toString(), validation: { ok: true } }
    fallback ??= normalized
  }
  return fallback
    ? { normalizedUrl: fallback.toString(), validation: { ok: true } }
    : { normalizedUrl: null, validation: createValidationError('invalid_url') }
}

export const validateFile = (file: File): ValidationResult => {
  if (!ACCEPTED_MIME_TYPES.has(file.type)) return createValidationError('invalid_type')
  if (file.size > MAX_UPLOAD_BYTES) return createValidationError('file_too_large')
  return { ok: true }
}
