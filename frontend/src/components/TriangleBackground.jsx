"use client"

import { useEffect, useState, useRef } from "react"
import PoissonDiskSampling from "poisson-disk-sampling"
import { Delaunay } from "d3-delaunay"

const COLORS = [" #ebebeb", "#f1f1f1", "#fefefe"]

export default function TriangleBackground({
  minDistance = 120,
  maxPoints = 300,
  edgeSpacing = 150,
  transitionDuration = 2000, // ms for color transition
  minInterval = 2000, // minimum ms between color changes
  maxInterval = 5000, // maximum ms between color changes
  driftAmount = 18, // maximum drift distance in pixels
  driftSpeed = 0.005, // speed of drift animation
}) {
  const [size, setSize] = useState(null)
  const [points, setPoints] = useState([])
  const [animatedPoints, setAnimatedPoints] = useState([])
  const [triangles, setTriangles] = useState([])
  const [triangleColors, setTriangleColors] = useState([])
  const animationRef = useRef(null)
  const timeRef = useRef(0)
  const staticPointsCountRef = useRef(0)

  useEffect(() => {
    const onResize = () => setSize({ width: window.innerWidth, height: window.innerHeight })
    onResize()
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])

  useEffect(() => {
    if (!size) return
    const { width, height } = size

    const pds = new PoissonDiskSampling({
      shape: [width, height],
      minDistance,
      maxDistance: minDistance * 1.8,
      tries: 25,
    })

    let pts = pds.fill()
    if (pts.length > maxPoints) {
      const step = Math.ceil(pts.length / maxPoints)
      pts = pts.filter((_, i) => i % step === 0)
    }

    const cornerPoints = [
      [0, 0],
      [width, 0],
      [0, height],
      [width, height],
    ]

    const edgePoints = []

    for (let x = edgeSpacing; x < width; x += edgeSpacing) {
      edgePoints.push([x, 0])
      edgePoints.push([x, height])
    }

    for (let y = edgeSpacing; y < height; y += edgeSpacing) {
      edgePoints.push([0, y])
      edgePoints.push([width, y])
    }

    const staticPointsCount = cornerPoints.length + edgePoints.length
    const allPoints = [...cornerPoints, ...edgePoints, ...pts]

    setPoints(allPoints)
    setAnimatedPoints(allPoints.map((p) => [...p]))

    const delaunay = Delaunay.from(allPoints)
    const tris = []
    for (let i = 0; i < delaunay.triangles.length; i += 3) {
      const a = delaunay.triangles[i]
      const b = delaunay.triangles[i + 1]
      const c = delaunay.triangles[i + 2]
      tris.push([a, b, c])
    }
    setTriangles(tris)

    setTriangleColors(tris.map(() => COLORS[Math.floor(Math.random() * COLORS.length)]))
  }, [size, minDistance, maxPoints, edgeSpacing])

  useEffect(() => {
    if (!size) return
    const { width, height } = size

    const cornerCount = 4
    const edgeCount = Math.floor(width / edgeSpacing) * 2 + Math.floor(height / edgeSpacing) * 2
    staticPointsCountRef.current = cornerCount + edgeCount
  }, [size, edgeSpacing])

  useEffect(() => {
    if (points.length === 0) return

    const animate = () => {
      timeRef.current += 1

      setAnimatedPoints((prevPoints) => {
        return points.map((originalPoint, index) => {
          if (index < staticPointsCountRef.current) {
            return originalPoint
          }

          const offsetX = Math.sin(timeRef.current * driftSpeed + index * 0.5) * driftAmount
          const offsetY = Math.cos(timeRef.current * driftSpeed + index * 0.7) * driftAmount

          return [originalPoint[0] + offsetX, originalPoint[1] + offsetY]
        })
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [points, driftAmount, driftSpeed])

  useEffect(() => {
    if (triangles.length === 0) return

    const timeouts = []

    triangles.forEach((_, index) => {
      const scheduleColorChange = () => {
        const randomInterval = minInterval + Math.random() * (maxInterval - minInterval)

        const timeout = setTimeout(() => {
          setTriangleColors((prev) => {
            const newColors = [...prev]
            let newColor
            do {
              newColor = COLORS[Math.floor(Math.random() * COLORS.length)]
            } while (newColor === prev[index] && COLORS.length > 1)
            newColors[index] = newColor
            return newColors
          })

          scheduleColorChange()
        }, randomInterval)

        timeouts.push(timeout)
      }

      const initialDelay = Math.random() * maxInterval
      const initialTimeout = setTimeout(scheduleColorChange, initialDelay)
      timeouts.push(initialTimeout)
    })

    return () => {
      timeouts.forEach((timeout) => clearTimeout(timeout))
    }
  }, [triangles, minInterval, maxInterval])

  if (!size || animatedPoints.length === 0 || triangles.length === 0) return null

  return (
    <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: -1 }}>
      <svg width={size.width} height={size.height} style={{ display: "block", width: "100vw", height: "100vh" }}>
        {triangles.map(([a, b, c], i) => {
          const pA = animatedPoints[a]
          const pB = animatedPoints[b]
          const pC = animatedPoints[c]
          if (!pA || !pB || !pC) return null

          const pointsStr = `${pA[0]},${pA[1]} ${pB[0]},${pB[1]} ${pC[0]},${pC[1]}`

          return (
          <polygon
            key={i}
            points={pointsStr}
            fill={triangleColors[i] || COLORS[0]}
            stroke={triangleColors[i] || COLORS[0]}  // same as fill
            strokeWidth={1} // optional
            style={{
              transition: `fill ${transitionDuration}ms ease-in-out, stroke ${transitionDuration}ms ease-in-out`,
            }}
          />

          )
        })}
      </svg>
    </div>
  )
}
