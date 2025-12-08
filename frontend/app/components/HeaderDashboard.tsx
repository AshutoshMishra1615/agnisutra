
import LanguageSwitcher from './LanguageSwitcher';
import Link from 'next/link';
import { Bell, User } from 'lucide-react';
import Image from 'next/image';
import { useTranslations } from 'next-intl';
import LogOut from './LogOut';

interface HeaderProps {
  userName?: string;
  showIcons?: boolean;
}

export default function Header({ userName, showIcons = true }: HeaderProps) {

  const t = useTranslations();

  return (
    <header className="bg-[rgba(255,255,255,0.12)] backdropb-blur-[7.2px] shadow-[0_2.4_6.78_rgba(0,0,0,0,76)] px-4 md:px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center space-x-2">
          <Image src="/images/logo-1.png" alt="logo" width={50} height={50} />
          <span className="text-white font-bold text-xl">{t('name')}</span>
        </Link>

        {showIcons && (
          <div className="flex items-center space-x-4">
            <button className="text-white hover:text-green-400 transition-colors">
              <Bell size={24} />
            </button>
            <button className="text-white hover:text-green-400 transition-colors">
              <User size={24} />
            </button>
            <LanguageSwitcher/>
            <LogOut/>
          </div>
        )}
      </div>
    </header>
  );
}
