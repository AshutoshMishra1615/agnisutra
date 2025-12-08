import {cookies} from 'next/headers';
import {getRequestConfig} from 'next-intl/server';

import enMessages from '../messages/en.json';
import hiMessages from '../messages/hi.json';
import mrMessages from '../messages/mr.json';

const messages = {
  en: enMessages,
  hi: hiMessages,
  mr: mrMessages,
};

 
export default getRequestConfig(async () => {
  const store = await cookies();
   const locale = (store.get('locale')?.value || 'en') as 'en' | 'hi';
 
  return {
    locale,
    messages: messages[locale] || messages.en
  };
});