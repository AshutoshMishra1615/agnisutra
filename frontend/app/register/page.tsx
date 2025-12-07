import RegisterCard from "../components/RegisterCard";

export default function Register() {
  return (
    <div className="lg:flex lg:justify-between lg:items-center lg:gap-20">
      <div className="hidden lg:block lg:flex-1">
        <div className=" flex flex-col justify-center px-16 space-y-6">
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            Maintain a digital farm diary
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            Monitor soil moisture levels
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            Track your crop's timeline
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            AI advisor for farming guidance
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl text-center">
            Real-time crop health insights
          </div>
         
        </div>
      </div>
      <div className="hidden lg:block border-2 border-white h-screen"></div>
      <div className="lg:flex-1">
        <RegisterCard />
      </div>
    </div>
  );
}
