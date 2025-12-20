import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

export const useWebSocket = (url: string, userId?: number) => {
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!url) return;

    const connect = () => {
      try {
        const socket = new WebSocket(url);

        socket.onopen = () => {
          console.log("WebSocket Connected");
          setIsConnected(true);
        };

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            // Check if the alert is for THIS user
            if (userId && data.user_id === userId) {
              const alerts = data.messages;
              if (Array.isArray(alerts)) {
                const alertText = alerts.join("\n");
                console.log("New Alert for Me:", alertText);

                toast.error("Critical Alert", {
                  description: alertText,
                  duration: 8000,
                  action: {
                    label: "Dismiss",
                    onClick: () => console.log("Alert dismissed"),
                  },
                });
              }
            }
          } catch (e) {
            console.error("Error processing WS message", e);
          }
        };

        socket.onclose = () => {
          console.log("WebSocket Disconnected");
          setIsConnected(false);
          // Reconnect after a delay
          setTimeout(connect, 3000);
        };

        socket.onerror = (error) => {
          console.error("WebSocket Error:", error);
          socket.close();
        };

        ws.current = socket;
      } catch (error) {
        console.error("WebSocket Connection Failed:", error);
        setTimeout(connect, 3000);
      }
    };

    connect();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [url, userId]);

  return { isConnected };
};
