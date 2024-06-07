import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const ChatUI = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const inputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    inputRef.current.style.height = "auto";
    inputRef.current.style.height = inputRef.current.scrollHeight + "px";
  }, [inputText]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (question) => {
    setIsProcessing(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/rag_api",
        {
          question,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.status >= 200 && response.status < 300) {
        const data = response.data;
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: data, sender: "bot" },
        ]);
      } else {
        throw new Error("Failed to send message");
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: inputText, sender: "user" },
    ]);
    sendMessage(inputText);
    setInputText("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleClear = () => {
    setMessages([]);
  };

  const scrollToBottom = () => {
    messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <>
      <div className="max-w-screen-md mx-auto my-8 p-6 bg-gray-100 rounded-lg shadow-lg">
        <div className="chat-messages">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${
                message.sender === "bot" ? "text-blue-600" : "text-green-600"
              } py-2`}
            >
              <ReactMarkdown>{message.text}</ReactMarkdown>
            </div>
          ))}
        </div>
        {messages.length > 0 && (
          <button
            onClick={handleClear}
            className="px-4 py-2 bg-red-500 text-white font-semibold rounded-lg"
          >
            Clear
          </button>
        )}
        <form onSubmit={handleSubmit} className="mt-4 flex">
          <textarea
            ref={inputRef}
            type="text"
            rows={2}
            readOnly={isProcessing}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={
              isProcessing ? "Please wait..." : "Type your message..."
            }
            className="flex-1 px-4 py-2 rounded-l-lg border border-gray-300 focus:outline-none focus:border-blue-500"
            style={{ minHeight: "4rem", resize: "none", overflowY: "hidden" }}
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white font-semibold rounded-r-lg"
          >
            Send
          </button>
          <div ref={messagesEndRef} />
        </form>
      </div>
    </>
  );
};

export default ChatUI;
