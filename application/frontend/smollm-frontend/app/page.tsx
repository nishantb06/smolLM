import { ButtonDemo } from "@/components/Button";
import { TextareaWithButton } from "@/components/TextAreaWithButton";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-2xl font-bold mb-4">SmolLM Frontend</h1>
      <div className="w-1/2">
        <TextareaWithButton />
      </div>
    </main>
  );
}
