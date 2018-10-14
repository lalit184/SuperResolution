from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch.optim as optim
import numpy as np
import pytorch_ssim_
from torch.autograd import Variable

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.pre_conv_1a=nn.Conv2d(3,80,3)
		self.pre_conv_1b=nn.Conv2d(80,160,3)
		self.dcn1_a=nn.Conv2d(160,160, 3)
		self.dcn1_b=nn.Conv2d(320,160, 3)
		self.dcn1_c=nn.Conv2d(480,160, 3)
		self.dcn1_d=nn.Conv2d(640,160, 3)
		self.dcn1_e=nn.Conv2d(800,160, 3)
		self.dcn1_f=nn.Conv2d(960,160, 3)
		self.dcn1_g=nn.Conv2d(1120,160, 3)
		self.dcn1_h=nn.Conv2d(1280,160, 3)
		self.dcn1_i=nn.Conv2d(1440,160, 3)

		self.x1ar=torch.nn.ReflectionPad2d(1)
		self.x1br=torch.nn.ReflectionPad2d(1)
		self.x1cr=torch.nn.ReflectionPad2d(1)
		self.x1dr=torch.nn.ReflectionPad2d(1)
		self.x1er=torch.nn.ReflectionPad2d(1)
		self.x1fr=torch.nn.ReflectionPad2d(1)
		self.x1gr=torch.nn.ReflectionPad2d(1)
		self.x1hr=torch.nn.ReflectionPad2d(1)
		self.x1ir=torch.nn.ReflectionPad2d(1)

		
		self.x2schuffle=nn.PixelShuffle(2)
		self.x4_input_schuffle=nn.PixelShuffle(2)

		self.pre_conv2=nn.Conv2d(40,160,5)
		
		self.dcn2_a=nn.Conv2d(160,160, 3)
		self.dcn2_b=nn.Conv2d(320,160, 3)
		self.dcn2_c=nn.Conv2d(480,160, 3)

		self.x2ar=torch.nn.ReflectionPad2d(1)
		self.x2br=torch.nn.ReflectionPad2d(1)
		self.x2cr=torch.nn.ReflectionPad2d(1)

		self.x4schuffle=nn.PixelShuffle(2)
		
		
		self.pre_conv3=nn.Conv2d(3,160,3)
		self.dcn3_a=nn.Conv2d(160,160, 3)

		self.u2_conv_a=nn.Conv2d(40,20,3)
		self.u2_conv_b=nn.Conv2d(20,3,5)
		self.x21r=torch.nn.ReflectionPad2d(1)
		self.x22r=torch.nn.ReflectionPad2d(2)
			
		self.u4_conv_a=nn.Conv2d(40,20,3)
		self.u4_conv_b=nn.Conv2d(20,3,5)
		self.x41r=torch.nn.ReflectionPad2d(1)
		self.x42r=torch.nn.ReflectionPad2d(2)

		self.x8_input_schuffle=nn.PixelShuffle(2)
				

		self.u8_conv_1a=nn.Conv2d(40,80,3)
		self.u8_conv_1b=nn.Conv2d(80,160,5)
		self.x81r=torch.nn.ReflectionPad2d(1)
		self.x82r=torch.nn.ReflectionPad2d(2)

		
		self.x8schuffle=nn.PixelShuffle(2)

		self.u8_conv_2a=nn.Conv2d(40,20,3)
		self.u8_conv_2b=nn.Conv2d(20,3,5)
		self.x81r_=torch.nn.ReflectionPad2d(1)
		self.x82r_=torch.nn.ReflectionPad2d(2)

		self.input_pad=torch.nn.ReflectionPad2d(1)
		self.x24_pad=torch.nn.ReflectionPad2d(1)
		self.x48_pad=torch.nn.ReflectionPad2d(2)
		

	def DCN9(self,x):

		x_=self.dcn1_a(x)
		x_=F.relu(x_)
		x_a=self.x1ar(x_)
		#print(x_a.shape)

		x_b=torch.cat((x,x_a),1)
		x_b=self.dcn1_b(x_b)
		x_b=F.relu(x_b)
		x_b=self.x1br(x_b)

		x_c=torch.cat((x,x_a,x_b),1)
		x_c=self.dcn1_c(x_c)
		x_c=F.relu(x_c)
		x_c=self.x1cr(x_c)
		#print(x_c.shape)

		x_d=torch.cat((x,x_a,x_b,x_c),1)
		x_d=self.dcn1_d(x_d)
		x_d=F.relu(x_d)
		x_d=self.x1dr(x_d)

		x_e=torch.cat((x,x_a,x_b,x_c,x_d),1)
		x_e=self.dcn1_e(x_e)
		x_e=F.relu(x_e)
		x_e=self.x1er(x_e)

		x_f=torch.cat((x,x_a,x_b,x_c,x_d,x_e),1)
		x_f=self.dcn1_f(x_f)
		x_f=F.relu(x_f)
		x_f=self.x1fr(x_f)

		x_g=torch.cat((x,x_a,x_b,x_c,x_d,x_e,x_f),1)
		x_g=self.dcn1_g(x_g)
		x_g=F.relu(x_g)
		x_g=self.x1gr(x_g)

		x_h=torch.cat((x,x_a,x_b,x_c,x_d,x_e,x_f,x_g),1)
		x_h=self.dcn1_h(x_h)
		x_h=F.relu(x_h)
		x_h=self.x1hr(x_h)

		x_i=torch.cat((x,x_a,x_b,x_c,x_d,x_e,x_f,x_g,x_h),1)
		x_i=self.dcn1_i(x_i)
		x_i=F.relu(x_i)
		dcn9=self.x1ir(x_i)

		return dcn9

	def DCN3(self,x):
		x_a_1=self.dcn2_a(x)
		x_a_1=F.relu(x_a_1)
		x_a_1=self.x2ar(x_a_1)
		
		x_b_1=torch.cat((x,x_a_1),1)
		x_b_1=self.dcn2_b(x_b_1)
		x_b_1=F.relu(x_b_1)
		x_b_1=self.x2br(x_b_1)

		x_c_1=torch.cat((x,x_a_1,x_b_1),1)
		x_c_1=self.dcn2_c(x_c_1)
		x_c_1=F.relu(x_c_1)
		dcn3=self.x2cr(x_c_1)

		return dcn3
		
			

	def X2(self,x):
		x=self.x2schuffle(x)
		x=self.u2_conv_a(x)
		x=F.relu(x)
		x=self.x21r(x)

		x=self.u2_conv_b(x)
		x=F.relu(x)
		x=self.x22r(x)

		return x


	def X4(self,x):
		x=self.x4schuffle(x)
		x=self.u4_conv_a(x)
		x=F.relu(x)
		x=self.x41r(x)

		x=self.u4_conv_b(x)
		x=F.relu(x)
		x=self.x42r(x)

		return x

	def X8(self,x):
		x=self.x8_input_schuffle(x)
		x=self.u8_conv_1a(x)
		x=F.relu(x)
		x=self.x81r(x)


		x=self.u8_conv_1b(x)
		x=F.relu(x)
		x=self.x82r(x)

		
		x=self.x8schuffle(x)
		x=self.u8_conv_2a(x)
		x=F.relu(x)
		x=x=self.x81r_(x)

		x=self.u8_conv_2b(x)
		x=F.relu(x)
		x=x=self.x81r_(x)

		return x




	def forward(self,x):
		#print("incoming",x.shape)
		
		x=self.pre_conv_1a(x)
		x=F.relu(x)
		x=self.input_pad(x)

		x=self.pre_conv_1b(x)

		x=F.relu(x)
		x=self.x24_pad(x)
		

		dcn9=self.DCN9(x)

		u2=self.X2(dcn9)

		

		x4i=self.x4_input_schuffle(dcn9)
		x4i=self.pre_conv2(x4i)
		
		x4i=F.relu(x4i)
		x4i=self.x48_pad(x4i)

		dcn3=self.DCN3(x4i)

		u4=self.X4(dcn3)

		u8=self.X8(dcn3)

		return u2,u4,u8




net=Net().double()
net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


def data(i):
	img=Image.open("./datasets/B100/HR/"+str(i+1)+".png")
	img=img.convert('RGB')
	
	basewidth,hsize=img.size
	

	img0 = img.resize((basewidth//8,hsize//8), Image.LANCZOS)
	img2 = img.resize((2*(basewidth//8),2*(hsize//8)), Image.LANCZOS)
	img4 = img.resize((4*(basewidth//8),4*(hsize//8)), Image.LANCZOS)
	img8 = img.resize((8*(basewidth//8),8*(hsize//8)), Image.LANCZOS)

	img0=np.transpose(np.array(img0),(2,0,1))
	img2=np.transpose(np.array(img2),(2,0,1))
	img4=np.transpose(np.array(img4),(2,0,1))
	img8=np.transpose(np.array(img8),(2,0,1))

	return np.array([img0],dtype=np.float64),np.array([img2],dtype=np.float64),np.array([img4],dtype=np.float64),np.array([img8],dtype=np.float64)

ssim_loss2 = pytorch_ssim_.SSIM(window_size = 5)
ssim_loss4 = pytorch_ssim_.SSIM(window_size = 10)
ssim_loss8 = pytorch_ssim_.SSIM(window_size = 20)

for epoch in range(50):  # loop over the dataset multiple times
	
	running_loss = 0.0
	for i in range(2945):
		# get the inputs
		u0,u2,u4,u8 = data(i)
		optimizer.zero_grad()

		u0=Variable(torch.from_numpy(u0),volatile=True).cuda()
		u2=Variable(torch.from_numpy(u2),volatile=True).cuda()
		u4=Variable(torch.from_numpy(u4),volatile=True).cuda()
		u8=Variable(torch.from_numpy(u8),volatile=True).cuda()
		

		o2,o4,o8 = net(u0).to("cpu",torch.double)
		loss =-16*ssim_loss2(o2,u2)+-4*ssim_loss2(o4,u4)+-1*ssim_loss2(o8,u8)
		loss.backward()
		optimizer.step()

		torch.cuda.empty_cache()



		o2n=o2.numpy()[0,:,:,:]
		o2n=np.transpose(o2n,(1,2,0))
		o2n=np.array(o2n,dtype=np.int8)
		img = Image.fromarray(o2n, 'RGB')
		img.save("./result2/"+str(i+1)+'.png')

		o2n=o4.numpy()[0,:,:,:]
		o2n=np.transpose(o2n,(1,2,0))
		o2n=np.array(o2n,dtype=np.int8)
		img = Image.fromarray(o2n, 'RGB')
		img.save("./result4/"+str(i+1)+'.png')

		o2n=o8.numpy()[0,:,:,:]
		o2n=np.transpose(o2n,(1,2,0))
		o2n=np.array(o2n,dtype=np.int8)
		img = Image.fromarray(o2n, 'RGB')
		img.save("./result8/"+str(i+1)+'.png')


		del u0
		del u2
		del u4
		del u8
		
		del o2
		del o4
		del o8
		
		# print statistics
		running_loss += loss.item()
		print(loss.item())
		
		if i % 800 == 0:    # print every 2000 mini-batches
			torch.save(net.state_dict(),"mymodel.pt")
			(epoch + 1, i + 1, running_loss / 800)
			running_loss = 0.0


print('Finished Training')



			