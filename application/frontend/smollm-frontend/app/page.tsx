import { TextareaWithButton } from "@/components/TextAreaWithButton";
import { TextareaWithLabel } from "@/components/TextAreaWithLabel";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-2xl font-bold mb-4">SmolLM Frontend</h1>
      <div className="w-1/2 space-y-4">
        <TextareaWithButton />
        <TextareaWithLabel />
      </div>
    </main>
  );
}
