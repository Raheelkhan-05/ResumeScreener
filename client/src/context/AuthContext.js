import { createContext, useContext, useState } from "react";
import api from "../api";

const AuthCtx = createContext(null);
export const useAuth = () => useContext(AuthCtx);

export default function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const token = localStorage.getItem("token");
    const email = localStorage.getItem("email");
    const username = localStorage.getItem("username");    
    return token && email ? { email, username } : null;
  });

  const login = async (email, password) => {
    const { data } = await api.post("/auth/login", { email, password });
    localStorage.setItem("token", data.token);
    localStorage.setItem("email", data.user.email);
    localStorage.setItem("username", data.user.username);
    setUser(data.user);
    console.log(user);
  };

  const signup = async (email, password, username) => {
  const { data } = await api.post("/auth/signup", { email, password, username });
  
  localStorage.setItem("token", data.token);
  localStorage.setItem("email", data.user.email);
  localStorage.setItem("username", data.user.username);

  setUser(data.user);
};


  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("email");
    localStorage.removeItem("username");
    setUser(null);
  };

  return (
    <AuthCtx.Provider value={{ user, login, signup, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}
