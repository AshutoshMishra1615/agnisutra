"use client";

import { useForm, SubmitHandler, SubmitErrorHandler } from "react-hook-form";
import { useTranslations } from "next-intl";
import axios from "axios";
import { toast } from 'sonner'
import { useRouter } from "next/navigation";

type FormValues = {
  name: String,
  email: String,
  password: String
}

export default function RegisterCard() {
  const {
    register,
    handleSubmit,
  } = useForm<FormValues>();

  const router = useRouter();

  const t = useTranslations();

  const onSubmit: SubmitHandler<FormValues> = async(data) => {
    try {
      const response = await axios.post('/auth/register',data);
      if(response.status === 200){
        //give a success message
        toast('Registration Successfull')
        //redirect to the dashboard
        router.push('/login')
      }
    } catch (error) {
      toast('Registration Unsuccessful')
      console.log(error)
    }
  }
  const onError: SubmitErrorHandler<FormValues> = (errors) =>
    console.log(errors)

  return (
    <div className="max-w-sm lg:max-w-lg px-7 py-7 rounded-[9.8px] border-[0.57px] border-white backdrop-blur-[6.79px] lg:rounded-[20px] lg:px-12 lg:py-12">
      <div>
        <p className="flex justify-center font-bold text-2xl lg:text-4xl pb-4">
          {t('register.title')}
        </p>
        <form
          onSubmit={handleSubmit(onSubmit, onError)}
          className="flex flex-col justify-start"
        >
          <label htmlFor="name" className="font-[600px] ">
            {t('register.name')}
          </label>
          <input
            type="text"
            id="name"
            placeholder={t('register.name')}
            {...register("name", { required: true })}
            className="w-full bg-white border-[0.5px] border-[#D0D5DD] rounded-sm shadow-[0_0.5_0.9_rgba(16,24,40,0.05) text-black px-3 py-1 text-sm my-3 "
          />
          
          <label htmlFor="email" className="font-[600px] ">
            {t('register.email')}
          </label>
          <input
            type="email"
            id="email"
            {...register("email", { required: true, pattern: {
    value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    message: t('register.invalidEmail')
  } })}
            placeholder={t('register.email')}
            className="w-full bg-white border-[0.5px] border-[#D0D5DD] rounded-sm shadow-[0_0.5_0.9_rgba(16,24,40,0.05) text-black px-3 py-1 text-sm my-3 "
          />
          
          <label htmlFor="password" className="font-[600px] ">
            {t('register.password')}
          </label>
          <input
            type="password"
            id="password"
            placeholder={t('register.password')}
            {...register("password", { required: true })}
            className="w-full bg-white border-[0.5px] border-[#D0D5DD] rounded-sm shadow-[0_0.5_0.9_rgba(16,24,40,0.05) text-black px-3 py-1 text-sm my-3 "
          />
          
          <div>
            <input type="checkbox" name="check" id="check" />
            <label htmlFor="check" className="pl-2 text-gray-300 text-[15px]" >
              {t('register.terms')}
            </label>
          </div>
          <button className="w-full bg-[#879d7b] px-2 py-1.5 rounded-sm shadow-[0_0.46_0.93_rgba(16,24,40,0.05)] font-bold text-[#132a13] my-4">{t('register.signup')}</button>
        </form>
        <p className="w-full flex text-gray-300 font-[500px] justify-center">
         {t('register.ask')} <span className="text-white underline"> {t('register.redirect')}</span>
        </p>
      </div>
    </div>
  );
}
