import userModel from "../models/usermodels.js"

export const getUserData=async (req,res)=>{
 try{
   const {userId}=req.body
   const user=await userModel.findOne({userId})
   if(!user){
    return res.json({success:false,message:"user not found"})
   }
   res.json({
    success:true,
    userData:{
      name:user.name,
      isAccountVerified:user.isAccountVerified
    }
   })
 }catch{
  return res.json({success:true,message:error.message})
 }
}
