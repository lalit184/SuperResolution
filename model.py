from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch.optim as optim
import numpy as np
import torchvision.datasets as datasets

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.pre_conv=nn.Conv2d(3,160,3)
		self.dcn1_a=nn.Conv2d(160,160, 3)
		self.dcn1_b=nn.Conv2d(320,160, 3)
		self.dcn1_c=nn.Conv2d(480,160, 3)
		self.dcn1_d=nn.Conv2d(640,160, 3)
		self.dcn1_e=nn.Conv2d(800,160, 3)
		self.dcn1_f=nn.Conv2d(960,160, 3)
		self.dcn1_g=nn.Conv2d(1120,160, 3)
		self.dcn1_h=nn.Conv2d(1280,160, 3)
		self.dcn1_i=nn.Conv2d(1440,160, 3)
		
		self.pre_conv2=nn.Conv2d(3,160,3)
		self.dcn2_a=nn.Conv2d(160,160, 3)
		self.dcn2_b=nn.Conv2d(320,160, 3)
		self.dcn2_c=nn.Conv2d(480,160, 3)
		
		self.pre_conv3=nn.Conv2d(3,160,3)
		self.dcn3_a=nn.Conv2d(160,160, 3)

		self.u2_conv_pre=nn.Conv2d(160,12,3)
		self.u4_conv_pre=nn.Conv2d(160,12,3)
		self.u8_conv_pre=nn.Conv2d(160,12,3)
		
		self.u2=nn.Conv2d(3,3,3)
		self.u4=nn.Conv2d(3,3,3)
		self.u8=nn.Conv2d(3,3,3)

	def forward(self,x):
		#print("incoming",x.shape)
	
		x=self.pre_conv(x)

		x=F.relu(x)
		x=torch.nn.ReflectionPad2d(1)(x)
		#print(x.shape)


		x_=self.dcn1_a(x)
		x_=F.relu(x_)
		x_a=torch.nn.ReflectionPad2d(1)(x_)
		#print(x_a.shape)

		x_b=torch.cat((x,x_a),1)
		x_b=self.dcn1_b(x_b)
		x_b=F.relu(x_b)
		x_b=torch.nn.ReflectionPad2d(1)(x_b)

		x_c=torch.cat((x,x_a,x_b),1)
		x_c=self.dcn1_c(x_c)
		x_c=F.relu(x_c)
		x_c=torch.nn.ReflectionPad2d(1)(x_c)
		#print(x_c.shape)

		x_d=torch.cat((x,x_a,x_b,x_c),1)
		x_d=self.dcn1_d(x_d)
		x_d=F.relu(x_d)
		x_d=torch.nn.ReflectionPad2d(1)(x_d)

		x_e=torch.cat((x,x_a,x_b,x_c,x_d),1)
		x_e=self.dcn1_e(x_e)
		x_e=F.relu(x_e)
		x_e=torch.nn.ReflectionPad2d(1)(x_e)

		x_f=torch.cat((x,x_a,x_b,x_c,x_d,x_e),1)
		x_f=self.dcn1_f(x_f)
		x_f=F.relu(x_f)
		x_f=torch.nn.ReflectionPad2d(1)(x_f)

		x_g=torch.cat((x,x_a,x_b,x_c,x_d,x_e,x_f),1)
		x_g=self.dcn1_g(x_g)
		x_g=F.relu(x_g)
		x_g=torch.nn.ReflectionPad2d(1)(x_g)

		x_h=torch.cat((x,x_a,x_b,x_c,x_d,x_e,x_f,x_g),1)
		x_h=self.dcn1_h(x_h)
		x_h=F.relu(x_h)
		x_h=torch.nn.ReflectionPad2d(1)(x_h)

		x_i=torch.cat((x,x_a,x_b,x_c,x_d,x_e,x_f,x_g,x_h),1)
		x_i=self.dcn1_i(x_i)
		x_i=F.relu(x_i)
		dcn9=torch.nn.ReflectionPad2d(1)(x_i)


		

		u2_=self.u2_conv_pre(dcn9)
		u2_=F.relu(u2_)
		u2_=torch.nn.ReflectionPad2d(1)(u2_)

		
		u2_pass=nn.PixelShuffle(2)(u2_)
		
		u2_=self.u2(u2_pass)
		u2=F.relu(u2_)
		u2=torch.nn.ReflectionPad2d(1)(u2)
		#print("result of 2x conv",u2.shape)
		

		u2_pass=self.pre_conv2(u2_pass)
		u2_pass=F.relu(u2_pass)
		u2_pass=torch.nn.ReflectionPad2d(1)(u2_pass)

		#print("layover input for 4s",u2_pass.shape)

		x_a_1=self.dcn2_a(u2_pass)
		x_a_1=F.relu(x_a_1)
		x_a_1=torch.nn.ReflectionPad2d(1)(x_a_1)
		
		x_b_1=torch.cat((u2_pass,x_a_1),1)
		x_b_1=self.dcn2_b(x_b_1)
		x_b_1=F.relu(x_b_1)
		x_b_1=torch.nn.ReflectionPad2d(1)(x_b_1)

		x_c_1=torch.cat((u2_pass,x_a_1,x_b_1),1)
		x_c_1=self.dcn2_c(x_c_1)
		x_c_1=F.relu(x_c_1)
		dcn3=torch.nn.ReflectionPad2d(1)(x_c_1)
		

		u4_=self.u4_conv_pre(dcn3)
		u4_=F.relu(u4_)
		u4_=torch.nn.ReflectionPad2d(1)(u4_)
		u4_pass=nn.PixelShuffle(2)(u4_)
		u4_=self.u4(u4_pass)
		u4=F.relu(u4_)
		u4=torch.nn.ReflectionPad2d(1)(u4)


		#print("result of 4x upsample",u4.shape)
		
		u4_pass=self.pre_conv3(u4_pass)
		u4_pass=F.relu(u4_pass)
		u4_pass=torch.nn.ReflectionPad2d(1)(u4_pass)

		#print("layover input for 8s",u4_pass.shape)



		x_a3=self.dcn3_a(u4_pass)
		x_a3=F.relu(x_a3)
		dcn1=torch.nn.ReflectionPad2d(1)(x_a3)

		u8_=self.u8_conv_pre(dcn1)
		u8_=F.relu(u8_)
		u8_=torch.nn.ReflectionPad2d(1)(u8_)
		u8_=nn.PixelShuffle(2)(u8_)
		u8_=self.u8(u8_)
		u8=F.relu(u8_)
		u8=torch.nn.ReflectionPad2d(1)(u8)

		#print("final result shape is ",u8.shape)
		
		

		return u2,u4,u8
net=Net().double()
#torch.nn.utils.clip_grad_norm(net.parameters(),)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


def data(i):
	img=Image.open("./HR/"+str(i+1)+".png")
	basewidth,hsize=img.size

	img0 = img.resize((basewidth//8,hsize//8), Image.ANTIALIAS)
	img2 = img.resize((2*(basewidth//8),2*(hsize//8)), Image.ANTIALIAS)
	img4 = img.resize((4*(basewidth//8),4*(hsize//8)), Image.ANTIALIAS)
	img8 = img.resize((8*(basewidth//8),8*(hsize//8)), Image.ANTIALIAS)

	img0=np.transpose(np.array(img0)[:,:,:3],(2,0,1))
	img2=np.transpose(np.array(img2)[:,:,:3],(2,0,1))
	img4=np.transpose(np.array(img4)[:,:,:3],(2,0,1))
	img8=np.transpose(np.array(img8)[:,:,:3],(2,0,1))

	return np.array([img0],dtype=np.float64),np.array([img2],dtype=np.float64),np.array([img4],dtype=np.float64),np.array([img8],dtype=np.float64)


for epoch in range(10):  # loop over the dataset multiple times
	
	running_loss = 0.0
	for i in range(100):
		# get the inputs
		u0,u2,u4,u8 = data(i)
		optimizer.zero_grad()

		u0=torch.from_numpy(u0)
		u0_=u0.type(torch.DoubleTensor)
		
		u2=torch.from_numpy(u2)
		u2_=u2.type(torch.DoubleTensor)
		
		u4=torch.from_numpy(u4)
		u4_=u4.type(torch.DoubleTensor)
		
		u8=torch.from_numpy(u8)
		u8_=u8.type(torch.DoubleTensor)
		
		
		# zero the parameter gradients
		

		# forward + backward + optimize
		o2,o4,o8 = net(u0_)



		loss =torch.nn.functional.mse_loss(o2,u2_)+torch.nn.functional.mse_loss(o4,u4_)+torch.nn.functional.mse_loss(o8,u8_)
		loss.backward()
		optimizer.step()
		
		# print statistics
		running_loss += loss.item()
		print(loss.item())
		
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0


print('Finished Training')



			