#

using Gadfly, Colors
using Distributions
using FFTW
using DSP



p1 = Normal(-pi/2, 0.5)
p2 = Normal(pi/2, 0.5)
p12_t = Normal(0.0, 1.0/sqrt(8.0))


x = range(-pi, stop=pi, length=1024)

g1 = (x)->pdf(p1,x)
g2 = (x)->pdf(p2,x)
g12_t = (x)->pdf(p12_t,x)


y1 = g1.(x)
y2 = g2.(x)
y12_t = g12_t.(x)

y12_t .*= 0.5
temp = fftshift(y12_t)
y12_t .+= temp


plot(y=y12_t, Geom.line)


Y12_t = fftshift(fft(y12_t))



pl = Gadfly.plot(
layer(y=abs.(Y1), Geom.line, Theme(default_color=colorant"red")),
layer(y=abs.(Y2), Geom.line, Theme(default_color=colorant"blue")),
layer(y=abs.(Y12_t), Geom.line, Theme(default_color=colorant"magenta"))
)

pl |> SVG("/tmp/test.svg",100cm,100cm)


y12_tr = ifft(ifftshift(Y12_t))

plot(y=abs.(y12_tr), Geom.line)



plot(
layer(x=x, y=y1, Geom.line, Theme(default_color=colorant"red")),
layer(x=x, y=y2, Geom.line, Theme(default_color=colorant"blue"))
)


using Makie


scene = lines(x, abs.(y12_tr), color = :blue)
# scatter!(scene, x, y1, color = :red, markersize = 0.1)





Y1 = fftshift(fft(y1))
Y2 = fftshift(fft(y2))
Y12_t = fft(y12_t)


plot(y=abs.(Y1), Geom.line)
plot(y=abs.(Y2), Geom.line)



Y12 = conv((Y1), (Y2))

st = 1
off = 2046
y12 = ifft(ifftshift(Y12[st:(st+off)]))


plot(y=abs.(y12), Geom.line)



# plot(x=real.(Y1), y=imag.(Y1), Geom.point)
# plot(x=real.(Y2), y=imag.(Y2), Geom.point)

plot(y=abs.(Y12), Geom.line)




##

#YY1 = [Y1; Y1; Y1]
YY2 = [ifftshift(Y2); ifftshift(Y2); ifftshift(Y2)]

plot(y=abs.(YY1), Geom.line)


YY12 = conv(ifftshift(Y1), (YY2))


plot(y=abs.(YY12), Geom.line)




st = 1024
off = 2047
yy12 = ifft(ifftshift(YY12[st:(st+off)]))


pl = plot(y=abs.(yy12), Geom.line)


##

using Cairo, Fontconfig
using Gadfly

pl |> PDF("/tmp/test.pdf", 30cm, 20cm)

run(`evince /tmp/test.pdf`)


##



plot(x=real.(Y12_t), y=imag.(Y12_t), Geom.point)

# brute force product
y12_d = y1 .* y2
y12_d ./= sum(y12_d)*(x[2]-x[1])



plot(
layer(x=x, y=y1, Geom.line, Theme(default_color=colorant"red")),
layer(x=x, y=y2, Geom.line, Theme(default_color=colorant"blue")),
layer(x=x, y=y12_d, Geom.line, Theme(default_color=colorant"magenta"))
)




# calculating the square using convolution

y1sqrt = sqrt.(y1)

Y1sqrt = fft(y1sqrt)

Y11 = conv(Y1sqrt, conj(Y1sqrt))

##

st = 1
off = 512
y11 = ifft(Y11[st:(st+off)])



plot(
layer(y=y1, Geom.line, Theme(default_color=colorant"red")),
layer(y=abs.(y11)./(sum(abs.(y11))*(x[2]-x[1])), Geom.line, Theme(default_color=colorant"blue"))
# layer(x=x, y=y12_d, Geom.line, Theme(default_color=colorant"magenta"))
)











## Linear convolution in Euclid space

xx = 0:127;

Y = (x::Real; mu::Real=0.0) -> pdf(Normal(mu, 1.0), x)

y1 = Y.(xx, mu=10)
y2 = Y.(xx, mu=20)

fY1 = fft(y1)
fY2 = fft(y2)

fYY = fY1.*fY2
yy = abs.(ifft(fYY))

plot(
  layer(
    x=xx,
    y=y1,
    Geom.line
  ),
  layer(
    x=xx,
    y=y2,
    Geom.line,
    Theme(default_color=colorant"deepskyblue")
  ),
  layer(
    x=xx,
    y=yy,
    Geom.line,
    Theme(default_color=colorant"magenta")
  )
)



## Test with Sine function

x = range(-7.5, stop=7.5, length=2048)
y = sin.(10.0*x) + sin.(100.0*x)

Y = fft(y)

Ys = fftshift(Y)

plot(y=abs.(Ys), Geom.line)



#
