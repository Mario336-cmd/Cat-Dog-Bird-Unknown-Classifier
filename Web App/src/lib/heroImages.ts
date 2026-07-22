export type HeroClass = 'cat' | 'dog' | 'bird'

const catModules = import.meta.glob('../assets/hero/cats/*.{jpg,jpeg,png,webp}', { eager: true, import: 'default' }) as Record<string, string>
const dogModules = import.meta.glob('../assets/hero/dogs/*.{jpg,jpeg,png,webp}', { eager: true, import: 'default' }) as Record<string, string>
const birdModules = import.meta.glob('../assets/hero/birds/*.{jpg,jpeg,png,webp}', { eager: true, import: 'default' }) as Record<string, string>

const sortedUrls = (modules: Record<string, string>) => Object.entries(modules)
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([, url]) => url)

const queues: Record<HeroClass, string[]> = {
  cat: sortedUrls(catModules),
  dog: sortedUrls(dogModules),
  bird: sortedUrls(birdModules),
}

const classes: HeroClass[] = ['cat', 'dog', 'bird']

const randomIndex = (length: number): number => {
  if (length <= 1) return 0
  const values = new Uint32Array(1)
  globalThis.crypto.getRandomValues(values)
  return values[0] % length
}

export const createHeroSequence = (): string[] => {
  const offsets: Record<HeroClass, number> = {
    cat: randomIndex(queues.cat.length),
    dog: randomIndex(queues.dog.length),
    bird: randomIndex(queues.bird.length),
  }
  const sequence: string[] = []
  const rounds = Math.max(...classes.map((name) => queues[name].length))
  for (let round = 0; round < rounds; round += 1) {
    for (const name of classes) {
      const images = queues[name]
      if (round < images.length && images.length) sequence.push(images[(offsets[name] + round) % images.length])
    }
  }
  return sequence
}

export const preloadHeroImages = (images: string[]): void => {
  images.forEach((url) => {
    const image = new Image()
    image.decoding = 'async'
    image.src = url
  })
}
