"use client";

import { useForm, SubmitHandler, SubmitErrorHandler } from "react-hook-form";
import { useTranslations } from "next-intl";
import axios from "axios";
import { toast } from 'sonner'
import { useRouter } from "next/navigation";
import { useAuth } from "../hooks/useAuth";

type FormValues = {
  username: String,
  password: String
}

export default function LoginCard() {
  const {
    register,
    handleSubmit,
  } = useForm<FormValues>();

  const t = useTranslations('login');

   const router = useRouter();

   const { setUser } = useAuth();

   const onSubmit: SubmitHandler<FormValues> = async(data) => {

    try {
      const response = await axios.post('/auth/login',data,{
       headers: {
         'Content-Type': "application/x-www-form-urlencoded"
       }
      });
      if(response.status === 200){
        const { access_Token } = response.data;
        const email = data.username
        setUser({ access_Token, email }); // Update user state in context
        axios.defaults.headers.common["Authorization"] = `Bearer ${access_Token}`; // Set default Authorization header
        toast.success("Login Successful");
        router.push("/dashboard"); // Redirect to dashboard
      }

    } catch (error) {
      toast.error("Login Unsuccessful");
      console.error("Login failed:", error);
    }
  }
  const onError: SubmitErrorHandler<FormValues> = (errors) =>
    console.log(errors)


  return (
    <div className="max-w-sm lg:max-w-lg px-7 py-7 rounded-[9.8px] border-[0.57px] border-white backdrop-blur-[6.79px] lg:rounded-[20px] lg:px-12 lg:py-12">
      <div>
        <p className="flex justify-center font-bold text-2xl lg:text-4xl pb-4">
          {t('title')}
        </p>
        <form
           onSubmit={handleSubmit(onSubmit, onError)}
          className="flex flex-col justify-start"
        >
           <label htmlFor="email" className="font-[600px] ">
            {t('email')}
          </label>
          <input
            type="email"
            id="email"
            {...register("username", { required: true})}
            placeholder={t('email')}
            className="w-full bg-white border-[0.5px] border-[#D0D5DD] rounded-sm shadow-[0_0.5_0.9_rgba(16,24,40,0.05) text-black px-3 py-1 text-sm my-3 "
          />
          
          <label htmlFor="password" className="font-[600px] ">
            {t('pass')}
          </label>
          <input
            type="password"
            id="password"
            placeholder={t('pass')}
            {...register("password", { required: true })}
            className="w-full bg-white border-[0.5px] border-[#D0D5DD] rounded-sm shadow-[0_0.5_0.9_rgba(16,24,40,0.05) text-black px-3 py-1 text-sm my-3 "
          />
          
          <button className="w-full bg-[#879d7b] px-2 py-1.5 rounded-sm shadow-[0_0.46_0.93_rgba(16,24,40,0.05)] font-bold text-[#132a13] my-4">{t('login')}</button>
        </form>
      </div>
    </div>
  );
}
