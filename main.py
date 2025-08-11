import numpy as np
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
plt.rcParams.update({'figure.max_open_warning': 0})

# ////////////////////////// Classes and functions /////////////////////////////////////////////////////////
class Event:
    #Initialize Event
    def __init__(self, active, reactive,ts, interval,time_counter):
        self.active = active
        self.reactive = reactive
        self.ts = ts
        self.toggle = True if active[ts]> active[ts-1] else False
        self.toggle_time = time_counter + ts
        self.deltap = np.zeros(interval)
        self.deltaq = np.zeros(interval)
        self.calc_active_and_reactive_power(active,reactive,ts,interval)
        self.cluster = -1

    def calc_active_and_reactive_power(self,active,reactive,ts,interval):
        if interval == 0 | interval == 1 | interval == 2:
            self.deltap[0] = abs(active[ts] - active[ts-1])
            self.deltaq[0] = abs(reactive[ts] - reactive[ts - 1])
            self.deltap[1] = abs(active[ts] - active[ts-1])
            self.deltaq[1] = abs(reactive[ts] - reactive[ts - 1])
            self.deltap[2] = abs(active[ts] - active[ts - 1])
            self.deltaq[2] = abs(reactive[ts] - reactive[ts - 1])
            self.deltap[3] = abs(active[ts] - active[ts - 1])
            self.deltaq[3] = abs(reactive[ts] - reactive[ts - 1])
        else:
            for i in range(interval):
                self.deltap[i] = abs(active[ts + i] - active[ts - interval + i])
                self.deltaq[i] = abs(reactive[ts + i] - reactive[ts - interval + i])

class Device:
    def __init__(self,name,event):
        self.events = []
        self.name = name
        self.events.append(event)
    def update_events_list(self,event):
        self.events.append(event)

def EventDetection(p_t):
    x, y = np.linspace(0, 1, p_t.size), p_t
    dx, dy = np.diff(x), np.diff(y)
    dir = dy / dx

    for i in range(dir.size):
        if abs(dir[i]) < 10:
            dir[i] = 0
    lst = []
    for i in range(dir.size):
        if abs(dir[i]) > 10:
            lst.append(i+1)
    return lst

def ExtractFeatures(p_t,q_t,ts_vec, time_counter):
    events = list()
    for i in range(len(ts_vec)):
        ts = ts_vec[i]
        if i == 0:
            tsprev = 0
        else:
            tsprev = ts_vec[i-1]
        if i == len(ts_vec) - 1:
            tsnext = len(p_t) - 1
        else:
            tsnext = ts_vec[i+1]
        interval = min(abs(ts - tsprev), abs(ts - tsnext), 50)
        events.append(Event(p_t, q_t, ts, interval,time_counter))
        #print("The event has a delta p of {} and delta q of {} and is a {} state".format(events[i].deltap,events[i].deltaq,"Toggle-On" if events[i].toggle == True else "Toggle - Off"))
    return events

def new_data_stream_classification(p_t_stream,q_t_stream, ts_vec, X, labels, num_of_clusters, time_counter):

    num_clusters = num_of_clusters
    if len(ts_vec) > 0:
        curr_events = ExtractFeatures(p_t_stream, q_t_stream, ts_vec,time_counter)
        deltaP_vec = []
        deltaQ_vec = []
        for event in curr_events:
            deltaP_vec.extend(np.absolute(event.deltap))
            deltaQ_vec.extend(np.absolute(event.deltaq))
        X_new = (deltaP_vec[1:], deltaQ_vec[1:])
        X_new = np.transpose(X_new)
        X = np.concatenate((X, X_new), axis=0) #Add to X Vector the X-Samples found in Data Stream
        brc.fit(X) #Re-Fit the new X Vector to Birch Clustering
        labels = brc.labels_
        num_clusters = len(np.unique(labels))
    else:
        curr_events = []
    return X, labels, num_clusters, curr_events

def CalcCostPerDevice(device):  # // This function calculates the power usage and cost of a curtain device
    power_usage_in_KWh= 0
    cost_per_KWh =0.4715
    i = 0
    while i < len(device.events) -1:
        if device.events[i].toggle and (not(device.events[i+1].toggle)):
            time_in_hours = (device.events[i+1].toggle_time - device.events[i].toggle_time)/3600
            power_usage_in_KWh += round(time_in_hours * device.events[i].deltap[1], 5)
            i += 2
    cost = round(power_usage_in_KWh *cost_per_KWh, 4)
    return power_usage_in_KWh, cost

def CalclTotalUsageAndCost(devices): #/// this function receives the on and off list of appliances and calculates the estimated cost
    total_power_usage_in_KWh = 0
    total_cost = 0
    for device in devices:
        power_usage_in_KWh, cost = CalcCostPerDevice(device)
        total_power_usage_in_KWh += power_usage_in_KWh
        total_cost += cost
    return round(total_power_usage_in_KWh, 4), round(total_cost, 4)


# //////////////////////////// Generating Data ////////////////////////////////////////////////////////
p_t = np.genfromtxt("KW.csv", delimiter=',', dtype=float)
q_t = np.genfromtxt("KVAR.csv", delimiter=',', dtype=float)

# ////////////////////////////Clustering whole chunk of data - Without labeling////////////////////////////////
'''                                            ---OPTIONAL---


events = list()
ts_vec = EventDetection(p_t)
events = ExtractFeatures(p_t, q_t, ts_vec)
deltaP_vec = []
deltaQ_vec = []
for event in events:
    deltaP_vec.extend(np.absolute(event.deltap))
    deltaQ_vec.extend(np.absolute(event.deltaq))
X = (deltaP_vec[1:], deltaQ_vec[1:])
X = np.transpose(X)
brc = Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None, threshold=0.5)
brc.fit(X)
color_theam = np.array(['red', 'green', 'blue', 'yellow, black'])
labels = brc.labels_
num_clusters = len(np.unique(labels))
plt.scatter(X[:, 0], X[:, 1], c=color_theam[labels], s=20)
plt.title('BIRCH :Estimated number of clusters: {}'.format(num_clusters))
plt.show()
'''

# /////////////////////////////// Clustering Data Stream /////////////////////////////////////////////////////////////
k = 0 #First Plot
time_counter = 0
event_counter =0
devices = []
clusters = [] #Initialize Cluster Array
num_clusters = 0 #Initialize Num of Clusters
startval = 30
#jump_val = math.floor((len(p_t) - startval)/segment)
jump_val = 21
tskip = 5
events = list() #Initialize Event List
ts_vec = EventDetection(p_t[0:startval])
events = ExtractFeatures(p_t[0:startval], q_t[0:startval], ts_vec,time_counter) #Extract Features for Events in 35 Samples
deltaP_vec = []
deltaQ_vec = []
for event in events:
    deltaP_vec.extend(np.absolute(event.deltap))
    deltaQ_vec.extend(np.absolute(event.deltaq))
X = (deltaP_vec[1:], deltaQ_vec[1:])
X = np.transpose(X)
brc = Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None, threshold=0.1) #Initialize Birch Settings
brc.fit(X) #Fit first 35 samples to Birch Clustering
labels = brc.labels_
n_clusters_ = len(np.unique(labels))
color_theam = np.array(['red', 'green', 'blue', 'yellow', 'purple'])
if (len(events) > 0) & (events != []):
    for event in events:
        event_counter += 1
        predict_arr = np.array([event.deltap[1:], event.deltaq[1:]])
        predict_arr = np.transpose(predict_arr)
        predict_vector = brc.predict(predict_arr)
        if predict_vector[0] > (len(clusters) - 1):
            new_device_name = input("NEW DEVICE DETECTED at {}: Please type name:".format(event.toggle_time))
            clusters.append(new_device_name)
            devices.append(Device(new_device_name,event))
        else:
            devices[predict_vector[0]].update_events_list(event)
        event.cluster = clusters[predict_vector[0]]
        if event.toggle:
            res = "On"
        else:
            res = "Off"
        print("The {} event is a {}-Type event that occurred at {}.It belongs to cluster #{}".format(event_counter, res,
                                                                                                     event.toggle_time,
                                                                                                     event.cluster))
time_counter +=startval
# ///////////////////////////////// Start Data Stream Clustering //////////////////////////////////////////
i = startval
while (i < len(p_t)):
    if i+jump_val > len(p_t):
        jump_val = len(p_t) - i - 1
    num_clusters_p = num_clusters
    plt.figure(k)
    k += 1
    plt.scatter(X[:, 0], X[:, 1], c=color_theam[labels], s=20)
    plt.title('BIRCH :Estimated number of clusters: {}'.format(num_clusters))
    plt.xlabel('Real power P [KW]')
    plt.ylabel('Reactive power Q [KVAR]')
    ts_vec = EventDetection(p_t[i:i + jump_val])
    if len(ts_vec) == 0:
        i += jump_val
        time_counter += jump_val
        continue
    flag = 0
    for j in range(len(ts_vec)):
        if (ts_vec[j] < tskip):
            time_counter -= tskip
            ts_vec = EventDetection(p_t[i-tskip:i - tskip + jump_val])

        if (ts_vec[j] > jump_val - tskip):
            jump_val += tskip
            ts_vec = EventDetection(p_t[i:i + jump_val])


    X, labels, num_clusters, temp_event = new_data_stream_classification(p_t[i:i + jump_val], q_t[i:i + jump_val],ts_vec,X, labels,num_clusters, time_counter)

    if (len(temp_event) > 0) & (temp_event != []):
        for event in temp_event:
            event_counter +=1
            predict_arr = np.array([event.deltap[1:], event.deltaq[1:]])
            predict_arr = np.transpose(predict_arr)
            predict_vector = brc.predict(predict_arr)
            if predict_vector[0] > (len(clusters)-1):
                new_device_name = input("NEW DEVICE DETECTED at {}: Please type name:".format(event.toggle_time))
                clusters.append(new_device_name)
                devices.append(Device(new_device_name, event))
            else:
                devices[predict_vector[0]].update_events_list(event)
            event.cluster = clusters[predict_vector[0]]
            if event.toggle:
                res = "On"
            else:
                res = "Off"
            print("The {} event is a {}-Type event that occurred at {}.It belongs to cluster #{}".format(event_counter, res, event.toggle_time, event.cluster))
        events.extend(temp_event)
    time_counter += jump_val
    i += jump_val
'''
print("Found {} Devices:".format(len(clusters)))
for i in range(len(clusters)):
    print("{}: {}".format(i,clusters[i]))
for i in range(len(events)):
 print("The {} event has a delta p of {} and delta q of {} and is a {} state, it belongs to {}".format(i, events[i].deltap,events[i].deltaq,"Toggle-On" if events[i].toggle == True else "Toggle - Off", labels[i+len(event.deltap)]))
'''
while True:
    print('What would you like to do next?')
    print('1)Check total cost and total energy usage')
    print('2)Check specific appliance usage')
    print('3)Check state of all appliance')
    print('4) exit')
    usr_inpt = input()
    if usr_inpt == '1':
        total_usage, total_cost = CalclTotalUsageAndCost(devices)
        print('Total usage until {} sec is {}KWh. the total cost is {}'.format(time_counter, total_usage, total_cost))
        print('')
    if usr_inpt == '2':
        for i in range(5):
            print('statistic on which device would you like to see?')
            flag = False
            usr_inpt2 = input()
            for device in devices:
                if device.name == usr_inpt2:
                    usage, temp = CalcCostPerDevice(device)
                    flag = True
                    break
            if flag == True:
                print('{} used {}KW ({}% of total power usage'.format(device.name, usage,round((usage * 100) / total_usage, 2)))
                print('')
                break
            else:
                print('No match found')
                print('')
                continue
    if usr_inpt == '3':
        for device in devices:
            if device.events[len(device.events)-1].toggle:
                res = "On"
            else:
                res = "Off"
            print('{} - {}'.format(device.name, res))
        print('')
    if usr_inpt == '4':
        break

# /////////////////////////////////////// Plots ///////////////////////////////////////////////////////////////////////

# Plot the event detection
fig = plt.figure(k)
k = k+1

gs = gridspec.GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs[0, :])
ax.plot(p_t)
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Real power P [KW]')

# Plot the event detection before smoothing
x, y = np.linspace(0, 1, p_t.size), p_t
dx, dy = np.diff(x), np.diff(y)
dir = dy / dx
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(dir)
plt.title('Event detection before smoothing')

# Plot the event detection after smoothing
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)
for i in range(dir.size):
    if abs(dir[i]) < 10:
        dir[i] = 0
ax3.plot(dir)
plt.title('Event detection after smoothing')

plt.figure(k)
plt.subplot(221)
plt.title("P(t)")
plt.xlabel('Time [sec]')
plt.ylabel('Real power P [KW]')
plt.plot(p_t)
plt.subplot(222)
plt.title("Q(t)")
plt.xlabel('Time [sec]')
plt.ylabel('Reactive power Q [KWAR]')
plt.plot(q_t)

plt.show()
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
