import React from "react";
import ChatUI from "./components/ChatUI";

const App = () => {
  return (
    <>
      <div className="sticky top-6 pt-6 pb-4 text-center font-bold text-3xl">
        Retrieval Augmented Generation (RAG)
      </div>
      <ChatUI />
    </>
  );
};

export default App;
