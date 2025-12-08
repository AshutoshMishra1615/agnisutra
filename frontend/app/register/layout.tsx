

import Image from "next/image";
import { useTranslations } from "next-intl";
import LanguageSwitcher from "../components/LanguageSwitcher";

export default function AuthLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {

  const t = useTranslations();

  return (
    <div className="bg-[url('/images/leaf-mobile.jpg')] h-screen md:bg-[url('/images/leaf-desktop.jpg')] bg-cover bg-no-repeat bg-fixed">
      <div className="px-2 py-1 mb-3 backdrop-blur-[7.2px] shadow-[0px_2.4px_6.78px_rgba(0,0,0,0.76)] md:backdrop-blur-md md:shadow-[0px_4px_11.3px_rgba(0,0,0,0.76)]  z-10">
        <div className="flex justify-between align-middle">
          <div className="flex gap-2 items-center">
            <Image src="/images/logo-1.png" alt="logo" width={50} height={50} />
            <span className="lemon-regular text-xl text-white md:text-2xl">{t('name')}</span>
          </div>
          <LanguageSwitcher/>
        </div>
      </div>
      <div className="flex items-center justify-center min-h-[90vh]">
        {children}
      </div>
    </div>
  );
}
