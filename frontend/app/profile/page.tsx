// app/profile/page.tsx
import type { User } from "../types/user"

// Example server-side data fetch. Replace with your real fetch to DB/API.
const getUser = async (): Promise<User> => {
  // Replace this with a call to your DB/API
  return {
    id: "TR64YE73",
    name: "Dipankar",
    email: "aniwat@gmail.com",
    mobile: "9485736746",
    username: "ani1232004",
    joinedOn: "2024-07-20",
    avatar: "/images/default-avatar.jpg",
    cover: "/images/default-cover.jpg",
  }
}

export default async function Page() {
  const user = await getUser()

  return (
   <></>
  )
}
