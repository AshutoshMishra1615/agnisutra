import LoginCard from "../components/LoginCard";
import { useTranslations } from "next-intl";


export default function Register() {

  const t = useTranslations(); 

  return (
    <div className="lg:flex lg:justify-between lg:items-center lg:gap-20">
      <div className="hidden lg:block lg:flex-1">
        <div className=" flex flex-col justify-center px-16 space-y-6">
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            {t('register.benefits.diary')}
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            {t('register.benefits.moisture')}
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            {t('register.benefits.timeline')}
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            {t('register.benefits.advisor')}
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            {t('register.benefits.health')}
          </div>
         
        </div>
      </div>
      <div className="hidden lg:block border-2 border-white h-screen"></div>
      <div className="lg:flex-1">
        <LoginCard />
      </div>
    </div>
  );
}
