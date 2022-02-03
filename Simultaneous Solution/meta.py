phi_temp = [];
phi_prime_temp = []
for N_temp in temp_1:
    phi_temp.append(Phi(N_temp))
    phi_prime_temp.append(Phi_Prime(N_temp))

plt.subplot(211)
plt.plot(temp_1, phi_temp)
plt.subplot(212)
plt.plot(temp_1, phi_prime_temp)
plt.show()