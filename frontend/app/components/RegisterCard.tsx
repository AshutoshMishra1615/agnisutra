"use client";

import { useForm, SubmitHandler, SubmitErrorHandler } from "react-hook-form";
import { useTranslations } from "next-intl";
import api from "../services/api";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { User, Mail, Lock, ArrowRight } from "lucide-react";

type FormValues = {
  name: string;
  email: string;
  password: string;
};

export default function RegisterCard() {
  const { register, handleSubmit } = useForm<FormValues>();

  const router = useRouter();

  const t = useTranslations();

  const onSubmit: SubmitHandler<FormValues> = async (data) => {
    try {
      const response = await api.post("/auth/register", data);
      if (response.status === 200) {
        //give a success message
        toast.success("Registration Successful");
        //redirect to the login
        router.push("/login");
      }
    } catch (error) {
      toast.error("Registration Unsuccessful");
      console.log(error);
    }
  };
  const onError: SubmitErrorHandler<FormValues> = (errors) =>
    console.log(errors);

  return (
    <div className="w-full max-w-md p-8 rounded-2xl bg-[#1a2e1a]/20 backdrop-blur-xl border border-[#879d7b]/20 shadow-2xl">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-white mb-2">
          {t("register.title")}
        </h2>
        <p className="text-gray-400 text-sm">
          Create your account to get started
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit, onError)} className="space-y-6">
        <div className="space-y-2">
          <label
            htmlFor="name"
            className="text-sm font-medium text-gray-300 block"
          >
            {t("register.name")}
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <User className="h-5 w-5 text-gray-500" />
            </div>
            <input
              type="text"
              id="name"
              placeholder={t("register.name")}
              {...register("name", { required: true })}
              className="w-full bg-[#050b05]/50 border border-[#879d7b]/30 text-white text-sm rounded-xl focus:ring-[#4ade80] focus:border-[#4ade80] block pl-10 p-3 placeholder-gray-500 transition-all"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="email"
            className="text-sm font-medium text-gray-300 block"
          >
            {t("register.email")}
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Mail className="h-5 w-5 text-gray-500" />
            </div>
            <input
              type="email"
              id="email"
              {...register("email", {
                required: true,
                pattern: {
                  value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                  message: t("register.invalidEmail"),
                },
              })}
              placeholder={t("register.email")}
              className="w-full bg-[#050b05]/50 border border-[#879d7b]/30 text-white text-sm rounded-xl focus:ring-[#4ade80] focus:border-[#4ade80] block pl-10 p-3 placeholder-gray-500 transition-all"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="password"
            className="text-sm font-medium text-gray-300 block"
          >
            {t("register.password")}
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Lock className="h-5 w-5 text-gray-500" />
            </div>
            <input
              type="password"
              id="password"
              placeholder={t("register.password")}
              {...register("password", { required: true })}
              className="w-full bg-[#050b05]/50 border border-[#879d7b]/30 text-white text-sm rounded-xl focus:ring-[#4ade80] focus:border-[#4ade80] block pl-10 p-3 placeholder-gray-500 transition-all"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            name="check"
            id="check"
            className="w-4 h-4 rounded border-gray-600 bg-[#050b05]/50 text-[#4ade80] focus:ring-[#4ade80]"
          />
          <label htmlFor="check" className="text-sm text-gray-400">
            {t("register.terms")}
          </label>
        </div>

        <button
          type="submit"
          className="w-full text-[#050b05] bg-[#4ade80] hover:bg-[#22c55e] focus:ring-4 focus:outline-none focus:ring-[#4ade80]/50 font-bold rounded-xl text-sm px-5 py-3 text-center flex items-center justify-center gap-2 transition-all transform hover:scale-[1.02]"
        >
          {t("register.signup")}
          <ArrowRight size={18} />
        </button>
      </form>

      <div className="mt-6 text-center">
        <p className="text-sm text-gray-400">
          {t("register.ask")}{" "}
          <a
            href="/login"
            className="text-[#4ade80] hover:text-[#22c55e] font-semibold hover:underline transition-colors"
          >
            Login here
          </a>
        </p>
      </div>
    </div>
  );
}
