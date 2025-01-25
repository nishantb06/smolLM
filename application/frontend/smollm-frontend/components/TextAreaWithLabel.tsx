"use client"

import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useState, useEffect } from "react"

export function TextareaWithLabel() {
  const [generatedText, setGeneratedText] = useState("");

  useEffect(() => {
    const handleTextGenerated = (event: CustomEvent<{ text: string }>) => {
      setGeneratedText(event.detail.text);
    };

    // Add event listener
    window.addEventListener("textGenerated", handleTextGenerated as EventListener);

    // Cleanup
    return () => {
      window.removeEventListener("textGenerated", handleTextGenerated as EventListener);
    };
  }, []);

  return (
    <div className="grid w-full gap-1.5">
      <Label htmlFor="message">Generated with SmolLM</Label>
      <Textarea 
        value={generatedText}
        placeholder="Output will be here..." 
        id="message" 
        readOnly
      />
    </div>
  )
}
