"use client";

import { useForm, SubmitHandler, SubmitErrorHandler } from "react-hook-form";
import { useTranslations } from "next-intl";
import api from "../services/api";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { useAuth } from "../hooks/useAuth";
import { Mail, Lock, ArrowRight } from "lucide-react";

type FormValues = {
  username: string;
  password: string;
};

export default function LoginCard() {
  const { register, handleSubmit } = useForm<FormValues>();

  const t = useTranslations("login");

  const router = useRouter();

  const { setUser } = useAuth();

  const onSubmit: SubmitHandler<FormValues> = async (data) => {
    try {
      // Backend expects form-data for OAuth2
      const formData = new FormData();
      formData.append("username", data.username as string);
      formData.append("password", data.password as string);

      const response = await api.post("/auth/login", formData, {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      });

      if (response.status === 200) {
        const { access_token } = response.data;
        const email = data.username;

        // Save token for API interceptor
        localStorage.setItem("token", access_token);

        // Update context (mapping to match existing context type if needed, or just use access_token)
        // Assuming AuthContext expects access_Token based on previous read
        setUser({ access_Token: access_token, email });

        toast.success("Login Successful");
        router.push("/dashboard"); // Redirect to dashboard
      }
    } catch (error) {
      toast.error("Login Unsuccessful");
      console.error("Login failed:", error);
    }
  };
  const onError: SubmitErrorHandler<FormValues> = (errors) =>
    console.log(errors);

  return (
    <div className="w-full max-w-md p-8 rounded-2xl bg-[#1a2e1a]/20 backdrop-blur-xl border border-[#879d7b]/20 shadow-2xl">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-white mb-2">{t("title")}</h2>
        <p className="text-gray-400 text-sm">Welcome back to AgniSutra</p>
      </div>

      <form onSubmit={handleSubmit(onSubmit, onError)} className="space-y-6">
        <div className="space-y-2">
          <label
            htmlFor="email"
            className="text-sm font-medium text-gray-300 block"
          >
            {t("email")}
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Mail className="h-5 w-5 text-gray-500" />
            </div>
            <input
              type="email"
              id="email"
              {...register("username", { required: true })}
              placeholder="name@example.com"
              className="w-full bg-[#050b05]/50 border border-[#879d7b]/30 text-white text-sm rounded-xl focus:ring-[#4ade80] focus:border-[#4ade80] block pl-10 p-3 placeholder-gray-500 transition-all"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="password"
            className="text-sm font-medium text-gray-300 block"
          >
            {t("pass")}
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Lock className="h-5 w-5 text-gray-500" />
            </div>
            <input
              type="password"
              id="password"
              placeholder="••••••••"
              {...register("password", { required: true })}
              className="w-full bg-[#050b05]/50 border border-[#879d7b]/30 text-white text-sm rounded-xl focus:ring-[#4ade80] focus:border-[#4ade80] block pl-10 p-3 placeholder-gray-500 transition-all"
            />
          </div>
        </div>

        <button
          type="submit"
          className="w-full text-[#050b05] bg-[#4ade80] hover:bg-[#22c55e] focus:ring-4 focus:outline-none focus:ring-[#4ade80]/50 font-bold rounded-xl text-sm px-5 py-3 text-center flex items-center justify-center gap-2 transition-all transform hover:scale-[1.02]"
        >
          {t("login")}
          <ArrowRight size={18} />
        </button>
      </form>

      <div className="mt-6 text-center">
        <p className="text-sm text-gray-400">
          Don&apos;t have an account?{" "}
          <a
            href="/register"
            className="text-[#4ade80] hover:text-[#22c55e] font-semibold hover:underline transition-colors"
          >
            Register here
          </a>
        </p>
      </div>
    </div>
  );
}
