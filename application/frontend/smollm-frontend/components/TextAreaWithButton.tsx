"use client"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"

// Update the interface to match the actual API response
interface GenerateResponse {
  output_text: string;  // Changed from 'text' to 'output_text'
}

export function TextareaWithButton() {
  const [prompt, setPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/text/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
          max_tokens: 100,
          temperature: 0.8,
          top_k: 40
        }),
      });

      const data: GenerateResponse = await response.json();
      
      // Update to use output_text instead of text
      const event = new CustomEvent("textGenerated", { 
        detail: { text: data.output_text } 
      });
      window.dispatchEvent(event);
    } catch (error) {
      console.error("Error generating text:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid w-full gap-2">
      <Textarea 
        placeholder="Type your prompt here." 
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <Button 
        onClick={handleGenerate}
        disabled={isLoading}
      >
        {isLoading ? "Generating..." : "Generate with SmolLM!"}
      </Button>
    </div>
  )
}