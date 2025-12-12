"use client"

import { useEffect, useRef, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"

import TriangleBackground from "./components/TriangleBackground.jsx"
import InputBar from "./components/InputBar.jsx"
import WelcomeMessage from "./components/WelcomeMessage.jsx"
import LLMMarkdownViewer from "./components/MarkdownViewer.jsx"
import UserMessageBubble from "./components/UserMessageBubble.jsx"
import CornerText from "./components/CornerText.jsx"
import LanguageSelector from "./components/LanguageSelector.jsx"
import LoadingAnimation from "./components/LoadingAnimation.jsx"
import ChatLoadingAnimation from "./components/ChatLoadingAnimation.jsx"

import circleCuoco from "./assets/circle_cuoco.png"

export default function Page() {

  const [query, setQuery] = useState("")
  const [chatHistory, setChatHistory] = useState([])
  const [uiState, setUiState] = useState("initial")
  const [promptQueue, setPromptQueue] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [lang, setLang] = useState(() => {
    try {
      const stored = typeof window !== "undefined" ? window.localStorage.getItem('cuoco_lang') : null
      if (stored === 'en' || stored === 'pt') return stored
    } catch {}
    return 'en'
  })
  const [showSharedInput, setShowSharedInput] = useState(false)

  const chatContainerRef = useRef(null)
  const bottomRef = useRef(null)     // ← sentinel ref for auto-scroll
  const inputRef = useRef(null)      // ← ref for input bar focus
  const processingRef = useRef(false)
  const retryTimeoutRef = useRef(null)
  const uiStateRef = useRef(uiState) // Ref to access current uiState in async functions

  // Keep uiStateRef in sync
  useEffect(() => {
    uiStateRef.current = uiState
  }, [uiState])

  /* -----------------------------------------------------
     FOCUS INPUT ON LOAD & STATE CHANGE
  ----------------------------------------------------- */
  useEffect(() => {
    if (uiState === 'initial' || showSharedInput) {
      // Small delay to ensure element is mounted/animated in
      setTimeout(() => {
        inputRef.current?.focus()
      }, 50)
    }
  }, [uiState, showSharedInput])

  /* -----------------------------------------------------
     CLEAR ON MOUNT (REFRESH)
  ----------------------------------------------------- */
  useEffect(() => {
    fetch("http://localhost:8000/clear", { method: "POST" }).catch(() => {})
  }, [])

  /* -----------------------------------------------------
     LANGUAGE STORAGE
  ----------------------------------------------------- */
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'cuoco_lang') {
        const v = e.newValue
        if (v === 'en' || v === 'pt') setLang(v)
      }
    }
    window.addEventListener('storage', handler)
    return () => window.removeEventListener('storage', handler)
  }, [])

  useEffect(() => {
    try {
      window.localStorage.setItem('cuoco_lang', lang)
    } catch {}
  }, [lang])



  /* -----------------------------------------------------
     QUEUE PROCESSING
  ----------------------------------------------------- */
  useEffect(() => {
    // Don't process if already processing or queue is empty
    if (processingRef.current || promptQueue.length === 0) return
    
    const currentItem = promptQueue[0]
    // Handle both object items (new format) and legacy strings
    const promptText = typeof currentItem === 'object' ? currentItem.text : currentItem
    const initialAddedToHistory = typeof currentItem === 'object' ? currentItem.addedToHistory : false

    const processPrompt = async (prompt, isAddedToHistory, retryDelay = 1000) => {
      // If we are in chat mode and the user message hasn't been shown yet, show it now
      // This handles the case where a message was queued during "loading" state but is processed after entering "chat" state
      let userMessageShown = isAddedToHistory
      if (!userMessageShown && uiStateRef.current === 'chat') {
        setChatHistory(prev => [...prev, { role: 'user', text: prompt }])
        userMessageShown = true
      }

      try {
        const res = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: prompt }),
        })

        const data = await res.json().catch(() => ({}))
        
        // Check if backend is not ready (503 or initialization error)
        const backendNotReady = !res.ok && (
          res.status === 503 || 
          (data?.error || "").toLowerCase().includes("initializ")
        )

        if (backendNotReady) {
          // Backend not ready, retry with exponential backoff
          const nextDelay = Math.min(retryDelay * 2, 10000)
          console.log(`[frontend] backend not ready, retrying in ${retryDelay}ms`)
          
          retryTimeoutRef.current = setTimeout(() => {
            processPrompt(prompt, userMessageShown, nextDelay)
          }, retryDelay)
          return
        }

        // Success
        const backendResponse = data.response || "No response"
        console.log(`[frontend] completed prompt:`, prompt)
        
        if (userMessageShown) {
          // User message already in history, just add bot response
          setChatHistory((prev) => [
            ...prev,
            { role: "bot", text: backendResponse }
          ])
        } else {
          // User message not in history (was processed during loading screen), add both
          setChatHistory((prev) => [
            ...prev,
            { role: "user", text: prompt },
            { role: "bot", text: backendResponse }
          ])
        }
        
        setUiState("chat")
        
        // Done processing this prompt - remove from queue
        processingRef.current = false
        setIsProcessing(false)
        setPromptQueue((prev) => prev.slice(1))
        
      } catch (err) {
        // Network error or other failure, retry with exponential backoff
        const nextDelay = Math.min(retryDelay * 2, 10000)
        console.log(`[frontend] error processing prompt, retrying in ${retryDelay}ms:`, err)
        
        retryTimeoutRef.current = setTimeout(() => {
          processPrompt(prompt, userMessageShown, nextDelay)
        }, retryDelay)
      }
    }
    
    // Start processing the first prompt in queue
    processingRef.current = true
    setIsProcessing(true)
    console.log(`[frontend] processing prompt:`, promptText, `| queue size:`, promptQueue.length)
    processPrompt(promptText, initialAddedToHistory)
    
  }, [promptQueue])

  /* -----------------------------------------------------
     SEND QUERY
  ----------------------------------------------------- */
  const handleKeyDown = async (e) => {
    if (e.key === "Enter") {
      await sendQuery()
    }
  }

  const sendQuery = async () => {
    if (!query.trim()) return
    
    const q = query.trim()
    setQuery("")
    
    // If in initial mode, switch to loading
    if (uiState === "initial") {
      setUiState("loading")
    }
    
    console.log(`[frontend] enqueuing prompt:`, q)
    // Store both text and whether it was already added to history
    setPromptQueue((prev) => [...prev, { text: q, addedToHistory: false }])
  }

  /* -----------------------------------------------------
     NEW CHAT (RESET)
  ----------------------------------------------------- */
  const handleNewChat = async () => {
    try {
      // clear backend conversation context
      await fetch("http://localhost:8000/clear", { method: "POST" })
        .catch(() => {})
    } catch {}

    // Clear any pending retry timeout
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current)
      retryTimeoutRef.current = null
    }

    // Reset all state
    processingRef.current = false
    setIsProcessing(false)
    setPromptQueue([])
    setChatHistory([])
    setQuery("")
    setUiState("initial")
    setShowSharedInput(false)

    // Force focus immediately (for case where we are already in initial state)
    setTimeout(() => {
      inputRef.current?.focus()
    }, 0)
  }


  /* -----------------------------------------------------
     ⭐ RELIABLE AUTO-SCROLL (NO CRASHES)
  ----------------------------------------------------- */
  useEffect(() => {
    if (!bottomRef.current) return

    // first attempt after paint
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "auto" })
    })

    // fallback for images / markdown expansion / animations
    const id = setTimeout(() => {
      bottomRef.current?.scrollIntoView({ behavior: "auto" })
    }, 80)

    return () => clearTimeout(id)

  }, [chatHistory])



  /* -----------------------------------------------------
     RENDER
  ----------------------------------------------------- */
  return (
    <div className="min-h-screen flex items-center justify-center relative">

      {/* background floating image */}
      <motion.img
        src={circleCuoco}
        alt="Cuoco circle"
        style={{ position: "fixed", top: -200, left: -256, width: 900, height: 900, zIndex: 10, rotate: "-20deg" }}
        initial={{ opacity: 0, y: -24 }}
        animate={{ opacity: 1, y: [ -24, 6, 0 ] }}
        transition={{ duration: 0.7, ease: "easeOut" }}
      />

      <TriangleBackground />

      <motion.div
        style={{ position: 'fixed', top: 16, right: 16, zIndex: 12 }}
        initial={{ opacity: 0, y: -24 }}
        animate={{ opacity: 1, y: [ -24, 6, 0 ] }}
        transition={{ delay: 0.15, duration: 0.7, ease: "easeOut" }}
      >
        <LanguageSelector lang={lang} onChange={setLang} />
      </motion.div>

      <div style={{ position: "fixed", left: 120, bottom: 100, zIndex: 11, pointerEvents: "none" }}>
        <CornerText lang={lang} align="left" gapPx={4} />
      </div>

      {/* MAIN CONTAINER */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
          width: "100%",
          position: "relative",
        }}
      >

        <AnimatePresence mode="wait" onExitComplete={() => {
          if (uiState !== 'initial') setShowSharedInput(true)
        }}>

          {/* INITIAL VIEW */}
          {uiState === "initial" && (
            <motion.div
              key="initial"
              initial={{ opacity: 0, y: -24 }}
              animate={{ opacity: 1, y: [ -24, 6, 0 ] }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.7, ease: "easeOut" }}
              style={{ display: "flex", flexDirection: "column", alignItems: "center", top:"20%" , left: "47%", position: "absolute" }}
              onAnimationComplete={() => inputRef.current?.focus()}
            >
              <WelcomeMessage className="mt-6 mb-4" lang={lang} />
              <div style={{ width: "850px", maxWidth: "90vw" }}>
                <InputBar
                  ref={inputRef}
                  placeholder={lang === 'pt' ? 'Pergunta ao Cuoco!' : 'Ask Cuoco!'}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onNewChat={handleNewChat}
                />
              </div>
            </motion.div>
          )}

          {/* LOADING VIEW */}
          {uiState === "loading" && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              style={{ textAlign: "center", fontSize: "24px", color: "#555" }}
            >
              <div style={{ width: "850px", maxWidth: "90vw", margin: "0 auto", left:"47%", position: "absolute", top: "45%" }}>
                <LoadingAnimation lang={lang} />
              </div>
            </motion.div>
          )}

          {/* CHAT VIEW */}
          {uiState === "chat" && (
            <motion.div
              key="chat"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}
            >
              <div
                ref={chatContainerRef}
                style={{
                  marginBottom: "20px",
                  width: "850px",
                  maxWidth: "90vw",
                  overflowY: "auto",
                  position: "absolute",
                  left: "47%",
                  top: "20%",
                  bottom: "12%"
                }}
              >
                {chatHistory.map((msg, i) =>
                  msg.role === "bot" ? (
                    <LLMMarkdownViewer key={i} text={msg.text} widthPx={850} />
                  ) : (
                    <div key={i} style={{ margin: "10px 0" }}>
                      <UserMessageBubble text={msg.text} maxWidthPx={Math.floor(850 * 2 / 3)} />
                    </div>
                  )
                )}

                {isProcessing && (
                  <div style={{ width: "850px", maxWidth: "90vw", display: "flex", justifyContent: "center" }}>
                    <ChatLoadingAnimation />
                  </div>
                )}

                {/* sentinel for auto-scroll */}
                <div ref={bottomRef} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* SHARED INPUT BAR */}
        <AnimatePresence>
          {showSharedInput && uiState !== "initial" && (
            <motion.div
              key="shared-input"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              style={{ position: "absolute", left: "47%", top: "90%" }}
              onAnimationComplete={() => inputRef.current?.focus()}
            >
              <div style={{ width: "850px", maxWidth: "90vw" }}>
                <InputBar
                  ref={inputRef}
                  placeholder={lang === 'pt' ? 'Pergunta ao Cuoco!' : 'Ask Cuoco!'}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onNewChat={handleNewChat}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
