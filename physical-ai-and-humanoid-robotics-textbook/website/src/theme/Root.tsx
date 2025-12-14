import React from 'react';
import ChatWidget from '../components/ChatWidget';

// Define the props type for Root component
interface RootProps {
  children: React.ReactNode;
}

// Default theme wrapper that injects the ChatWidget
export default function Root({ children }: RootProps) {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
}